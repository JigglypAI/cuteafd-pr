#!/usr/bin/env python3
"""Compare the engine PE owner observers to the qualified composed AOT path.

Run inside the matching dev image. CPU checkpoint tensors are uploaded directly
into one packed native arena; the composed replay streams one layer at a time.
"""
from __future__ import annotations
import argparse
import ctypes as C
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

COMMON = Path(__file__).resolve().parents[1] / "mimo_v2/qualify-vision.py"
module = importlib.util.spec_from_file_location("vision_common", COMMON)
common = importlib.util.module_from_spec(module)
module.loader.exec_module(common)
NO = common.NO_OFFSET


def pack(kind, tensors, max_tokens):
    ink = kind == "inkling2"
    h, intermediate, depth, patch = (1024,4096,23,20) if ink else (1536,8960,47,14)
    prefix = "model.visual.backbone" if ink else "vision_model"
    spec = common.Spec(abi_version=3,reserved=6 if ink else 7,max_tokens=max_tokens,
        output_width=4608 if ink else 4096,hidden=h,depth=depth,heads=16,kv_heads=16,
        head_dim=h//16,intermediate=intermediate,patch_size=patch,merger_width=4*h,norm_eps=1e-5)
    extension = common.PerceptionSpec(activation=int(not ink),position_grid=32 if ink else 52,
        class_token=int(ink),output_kind=int(not ink))
    scales = [(C.c_uint64*depth)() for _ in range(2)]
    extension.scale1,extension.scale2 = scales
    spec.perception=C.pointer(extension)
    spec._pe_storage=(extension,scales)
    uploads=[];offsets={};cursor=0
    def put(name):
        nonlocal cursor
        value=tensors[name]
        assert value.dtype==torch.bfloat16 and value.is_contiguous(),name
        cursor=(cursor+255)//256*256
        offsets[name]=cursor;uploads.append((cursor,value));cursor+=value.numel()*2
        return offsets[name]
    for field in ("patch_bias","merger_norm","merger_norm_bias","merger_fc1","merger_fc2","merger_fc1_bias","merger_fc2_bias"):
        setattr(spec,field,NO)
    spec.merger_extra[:]=[NO]*8
    spec.patch=put(prefix+(".embeddings.patch_embedding.weight" if ink else ".conv1.weight"))
    spec.pos_embed=put(prefix+(".embeddings.position_embedding" if ink else ".positional_embedding"))
    if ink:
        extension.class_embedding=put(prefix+".embeddings.class_embedding")
        extension.class_position=put(prefix+".embeddings.cls_position_embedding")
    else:extension.class_embedding=extension.class_position=NO
    extension.ln_pre=put(prefix+".ln_pre.weight");extension.ln_pre_bias=put(prefix+".ln_pre.bias")
    for index in range(depth):
        base=prefix+(f".layers.{index}." if ink else f".transformer.resblocks.{index}.")
        block=spec.blocks[index]
        if ink:
            weights=[put(base+f"self_attn.{part}_proj.weight") for part in "qkv"]
            biases=[put(base+f"self_attn.{part}_proj.bias") for part in "qkv"]
            assert weights==[weights[0]+i*h*h*2 for i in range(3)]
            assert biases==[biases[0]+i*h*2 for i in range(3)]
            block.qkv,block.qkv_bias=weights[0],biases[0]
        else:block.qkv=put(base+"attn.in_proj_weight");block.qkv_bias=put(base+"attn.in_proj_bias")
        attn="self_attn" if ink else "attn"
        for field,suffix in [("proj",attn+".out_proj.weight"),("proj_bias",attn+".out_proj.bias"),
            ("gate_up","mlp.c_fc.weight"),("gate_up_bias","mlp.c_fc.bias"),
            ("down","mlp.c_proj.weight"),("down_bias","mlp.c_proj.bias")]:setattr(block,field,put(base+suffix))
        for suffix in ("ln_1.weight","ln_1.bias","ln_2.weight","ln_2.bias","ls_1.gamma","ls_2.gamma"):put(base+suffix)
        block.norm1=offsets[base+"ln_1.weight"];block.norm2=offsets[base+"ln_2.weight"]
        spec.norm1_bias[index]=offsets[base+"ln_1.bias"];spec.norm2_bias[index]=offsets[base+"ln_2.bias"]
        scales[0][index]=offsets[base+"ls_1.gamma"];scales[1][index]=offsets[base+"ls_2.gamma"]
        block.key0_bias=spec.q_norm[index]=spec.k_norm[index]=NO
    extension.downsample1=extension.downsample1_bias=extension.downsample2=extension.downsample2_bias=NO
    if ink:
        spec.merger_fc1=put("model.visual.proj_in.weight");spec.merger_fc2=put("model.visual.proj_out.weight")
    else:
        for field,suffix in [("downsample1","vit_downsampler1.weight"),("downsample1_bias","vit_downsampler1.bias"),
            ("downsample2","vit_downsampler2.weight"),("downsample2_bias","vit_downsampler2.bias")]:setattr(extension,field,put(prefix+"."+suffix))
        spec.merger_fc2=put("vit_large_projector.weight")
    assert set(offsets)==set(tensors),set(tensors)-set(offsets)
    source=Path(__file__).resolve().parents[4]/"rust/crates/cuteafd-loader/src/media/perception_vision.rs"
    import re
    text=source.read_text().split("impl PerceptionKind {",1)[1]
    key="Inkling2" if ink else "Step5"
    bits=re.search(r"PerceptionKind::"+key+r" => &\[([^]]+)\]",text)[1]
    frequencies=torch.from_numpy(np.array([int(value,16) for value in re.findall(r"0x[0-9a-f]+",bits)],dtype=np.uint32).view(np.float32).copy())
    spec.inv_freq=(cursor+255)//256*256;spec.weight_bytes=spec.inv_freq+frequencies.numel()*4
    uploads.append((spec.inv_freq,frequencies))
    return spec,uploads,offsets


def check(status):
    if status:raise RuntimeError(f"native CUDA status {status}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind",choices=("inkling2","step5"),required=True)
    parser.add_argument("--snapshot",type=Path,required=True)
    parser.add_argument("--library",type=Path,required=True)
    parser.add_argument("--image",type=Path,required=True)
    parser.add_argument("--side",type=int,choices=(504,728),default=504)
    parser.add_argument("--preflight-only",action="store_true")
    args=parser.parse_args()
    from safetensors import safe_open
    from PIL import Image
    ink=args.kind=="inkling2"
    index=json.loads((args.snapshot/"model.safetensors.index.json").read_text())["weight_map"]
    prefixes=("model.visual.",) if ink else ("vision_model.","vit_large_projector.")
    tensors={}
    for shard in sorted({value for key,value in index.items() if key.startswith(prefixes)}):
        with safe_open(args.snapshot/shard,framework="pt",device="cpu") as file:
            for name in file.keys():
                if name.startswith(prefixes):tensors[name]=file.get_tensor(name)
    assert len(tensors)==(422 if ink else 667)
    spec,uploads,offsets=pack(args.kind,tensors,169)
    raw=Image.open(args.image).convert("RGB").crop((0,0,197,157) if ink else (0,0,504,504))
    if ink:
        rgb=np.zeros((160,200,3),dtype=np.uint8);rgb[:157,:197]=np.array(raw);gh,gw=8,10;content=(0,0,197,157)
    else:rgb=np.array(raw);gh=gw=args.side//14;content=(0,0,rgb.shape[1],rgb.shape[0])
    mean=np.array([.48145466,.4578275,.40821073],dtype=np.float32)
    std=np.array([.26862954,.26130258,.27577711],dtype=np.float32)
    processor_std=np.array([.26862954,.2613026,.2757771],dtype=np.float32)
    parameters=np.zeros(768,dtype=np.float32);parameters[:3]=mean;parameters[3:6]=processor_std if ink else std;parameters[6:9]=std
    library=C.CDLL(str(args.library));ptr=C.c_void_p;u64=C.c_uint64;i32=C.c_int32
    library.cuteafd_vision_required.argtypes=[C.POINTER(common.Spec),C.POINTER(common.Ledger)]
    class Tags(C.Structure):
        _fields_=[("weight_kind",C.c_uint32),("weight_owner",C.c_char_p),("workspace_kind",C.c_uint32),("workspace_owner",C.c_char_p)]
    tags=Tags(1,b"vision/pe-oracle/weights",9,b"vision/pe-oracle/workspace")
    library.cuteafd_vision_create.argtypes=[C.POINTER(common.Spec),i32,u64,C.POINTER(Tags),C.POINTER(ptr)]
    library.cuteafd_vision_weight_arena.argtypes=[ptr,C.POINTER(ptr)]
    library.cuteafd_vision_commit_weights.argtypes=[ptr]
    library.cuteafd_vision_upload.argtypes=[ptr,u64,ptr,u64]
    library.cuteafd_vision_encode_pixels.argtypes=[ptr,ptr,u64,ptr]+[i32]*8+[ptr,u64,common.OBSERVER,ptr]
    library.cuteafd_vision_destroy.argtypes=[ptr]
    library.cuteafd_vision_get_ledger.argtypes=[ptr,C.POINTER(common.Ledger)]
    ledger=common.Ledger();check(library.cuteafd_vision_required(C.byref(spec),C.byref(ledger)))
    admitted=ledger.weights+ledger.scratch+ledger.blas_workspace
    assert ledger.weights==spec.weight_bytes and ledger.scratch>0 and ledger.blas_workspace==0
    invalid=common.Ledger();original=spec.max_tokens
    spec.max_tokens=0;assert library.cuteafd_vision_required(C.byref(spec),C.byref(invalid))!=0
    spec.max_tokens=4097;assert library.cuteafd_vision_required(C.byref(spec),C.byref(invalid))!=0
    spec.max_tokens=original
    if args.preflight_only:
        print("PE_OWNER_PREFLIGHT "+json.dumps(dict(kind=args.kind,weights=ledger.weights,scratch=ledger.scratch)),flush=True)
        return
    owner=ptr()
    assert library.cuteafd_vision_create(C.byref(spec),0,admitted-1,C.byref(tags),C.byref(owner))!=0 and not owner.value
    check(library.cuteafd_vision_create(C.byref(spec),0,admitted,C.byref(tags),C.byref(owner)))
    stages={};failures=[]
    @common.OBSERVER
    def observe(ctx,stage,data,rows,width,column):
        try:
            assert not column
            value=torch.empty(rows,width,device="cuda",dtype=torch.bfloat16)
            from cuda.bindings import runtime
            check(int(runtime.cudaMemcpy(value.data_ptr(),data,value.numel()*2,runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice)[0]))
            stages[stage]=value.cpu()
            return 0
        except Exception as error:failures.append(repr(error));return 1
    units=gh*gw//(4 if ink else 16)
    output=np.empty((units,spec.output_width),dtype=np.uint16)
    try:
        arena=ptr();check(library.cuteafd_vision_weight_arena(owner,C.byref(arena)))
        for offset,value in uploads:check(library.cuteafd_vision_upload(owner,offset,value.data_ptr(),value.numel()*value.element_size()))
        check(library.cuteafd_vision_commit_weights(owner))
        check(library.cuteafd_vision_encode_pixels(owner,rgb.ctypes.data,rgb.nbytes,parameters.ctypes.data,
            gh,gw,rgb.shape[0],rgb.shape[1],*content,output.ctypes.data,output.nbytes,observe,None))
        final_ledger=common.Ledger();check(library.cuteafd_vision_get_ledger(owner,C.byref(final_ledger)))
        assert final_ledger.device_allocations==2 and final_ledger.encodes==1
        assert (final_ledger.weights,final_ledger.scratch)==(ledger.weights,ledger.scratch)
    finally:check(library.cuteafd_vision_destroy(owner))
    assert not failures,failures
    print("PE_OWNER_LEDGER "+json.dumps(ledger.as_dict()),flush=True)
    replay(args,tensors,uploads[-1][1],rgb,parameters,gh,gw,content,stages)


def replay(args,tensors,frequencies,rgb,parameters,gh,gw,content,stages):
    from b12x.integration.cuteafd import media,perception,perception_forward,image_resize,perception_pixels
    from cuda.bindings import runtime
    ink=args.kind=="inkling2";h,intermediate,depth,patch=(1024,4096,23,20) if ink else (1536,8960,47,14)
    prefix="model.visual.backbone" if ink else "vision_model"
    n=gh*gw+int(ink);units=gh*gw//(4 if ink else 16)
    def tensor(name):return tensors[name].cuda()
    def compare(stage,value):
        actual=value.cpu();expected=stages[stage]
        equal=torch.equal(actual,expected)
        print("PE_OWNER_BOUNDARY "+json.dumps(dict(stage=stage,byte_exact=equal,rows=actual.shape[0])),flush=True)
        assert equal,f"native/composed stage {stage} differs"
    device_rgb=torch.from_numpy(rgb).cuda();params=torch.from_numpy(parameters).cuda()
    patches=torch.empty(gh*gw,3*patch*patch,device="cuda",dtype=torch.bfloat16)
    if ink:perception_pixels.compile_pixel_patches_aot().launch(device_rgb,params[:3],params[3:6],params[6:9],patches,scalars=[*rgb.shape[:2],*content])
    else:
        pixels=torch.empty(*rgb.shape,device="cuda")
        image_resize.compile_rgb_normalize_aot().launch(device_rgb,params[:3],params[3:6],pixels,scalars=list(rgb.shape[:2]))
        side=args.side
        coefficients=image_resize.compile_resize_coefficients_aot(filter="bilinear")
        def coeff(source):
            taps=image_resize.resize_max_taps(source,side,filter="bilinear")
            indices=torch.empty(side,device="cuda",dtype=torch.int32);sizes=torch.empty_like(indices);weights=torch.empty(side,taps,device="cuda")
            coefficients.launch(indices,sizes,weights,scalars=[source,side,taps]);return indices,sizes,weights,taps
        ix,nx,wx,tx=coeff(rgb.shape[1]);iy,ny,wy,ty=coeff(rgb.shape[0])
        horizontal=torch.empty(rgb.shape[0],side,3,device="cuda");resized=torch.empty(side,side,3,device="cuda")
        image_resize.compile_resize_pass_aot(vertical=False).launch(pixels,ix,nx,wx,horizontal,scalars=[*rgb.shape[:2],side,tx])
        image_resize.compile_resize_pass_aot(vertical=True).launch(horizontal,iy,ny,wy,resized,scalars=[rgb.shape[0],side,side,ty])
        perception_forward.compile_normalized_patches_aot(patch=14).launch(resized,patches,scalars=[side,side])
    patch_emb=torch.empty(gh*gw,h,device="cuda",dtype=torch.bfloat16)
    weight=tensor(prefix+(".embeddings.patch_embedding.weight" if ink else ".conv1.weight"))
    media.compile_patch_projection_aot(width=3*patch*patch,outputs=h).launch(patches,weight,patch_emb,scalars=[gh*gw])
    position=tensor(prefix+(".embeddings.position_embedding" if ink else ".positional_embedding"))
    dummy=torch.zeros(h,device="cuda",dtype=torch.bfloat16)
    cls=tensor(prefix+".embeddings.class_embedding") if ink else dummy
    cls_position=tensor(prefix+".embeddings.cls_position_embedding") if ink else dummy
    x=torch.empty(n,h,device="cuda",dtype=torch.bfloat16);tmp=torch.empty_like(x);normed=torch.empty_like(x)
    norm=perception_forward.compile_layer_norm_aot(hidden=h,vendor_rsqrt=ink)
    perception_forward.compile_learned_position_aot(hidden=h,grid=32 if ink else 52,class_token=ink,torch_bilinear=not ink).launch(patch_emb,position,cls,cls_position,tmp,scalars=[gh,gw])
    norm.launch(tmp,tensor(prefix+".ln_pre.weight"),tensor(prefix+".ln_pre.bias"),x,scalars=[n,1e-5]);compare(0,x)
    qkv=torch.empty(n,3*h,device="cuda",dtype=torch.bfloat16)
    q,k,v,o=[torch.empty(16384,16,h//16,device="cuda",dtype=torch.bfloat16) for _ in range(4)]
    cu=torch.tensor([0,n],device="cuda",dtype=torch.int32);lse=torch.empty(16,16384,device="cuda")
    freq=frequencies.cuda();status,l2=runtime.cudaDeviceGetAttribute(runtime.cudaDeviceAttr.cudaDevAttrL2CacheSize,0);check(int(status))
    rotary=perception.compile_rotary_2d_aot(heads=16,dim=h//16,class_token=ink)
    attention=media.compile_vision_attention_aot(dim=h//16)
    qproj=media.compile_biased_projection_aot(width=h,outputs=3*h);oproj=media.compile_biased_projection_aot(width=h,outputs=h)
    up=media.compile_biased_projection_aot(width=h,outputs=intermediate);down=media.compile_biased_projection_aot(width=intermediate,outputs=h)
    residual=perception.compile_scaled_residual_aot(width=h)
    activation=media.compile_gelu_aot(width=intermediate) if ink else perception.compile_quick_gelu_aot(width=intermediate)
    delta=torch.empty_like(x);middle=torch.empty_like(x);mlp=torch.empty(n,intermediate,device="cuda",dtype=torch.bfloat16)
    for index in range(depth):
        base=prefix+(f".layers.{index}." if ink else f".transformer.resblocks.{index}.")
        weights={name[len(base):]:value.cuda() for name,value in tensors.items() if name.startswith(base)}
        attn="self_attn" if ink else "attn"
        qw=torch.cat([weights[f"self_attn.{p}_proj.weight"] for p in "qkv"]) if ink else weights["attn.in_proj_weight"]
        qb=torch.cat([weights[f"self_attn.{p}_proj.bias"] for p in "qkv"]) if ink else weights["attn.in_proj_bias"]
        norm.launch(x,weights["ln_1.weight"],weights["ln_1.bias"],normed,scalars=[n,1e-5])
        qproj.launch(normed,qw,qb,qkv,scalars=[n]);rotary.launch(qkv,freq,q,k,v,scalars=[n,gw])
        attention.launch(q,k,v,o,lse,cu,scalars=[(h//16)**-.5,l2])
        oproj.launch(o,weights[attn+".out_proj.weight"],weights[attn+".out_proj.bias"],delta,scalars=[n])
        residual.launch(x,delta,weights["ls_1.gamma"],middle,scalars=[n])
        norm.launch(middle,weights["ln_2.weight"],weights["ln_2.bias"],normed,scalars=[n,1e-5])
        up.launch(normed,weights["mlp.c_fc.weight"],weights["mlp.c_fc.bias"],mlp,scalars=[n]);activation.launch(mlp,mlp,scalars=[n])
        down.launch(mlp,weights["mlp.c_proj.weight"],weights["mlp.c_proj.bias"],delta,scalars=[n])
        residual.launch(middle,delta,weights["ls_2.gamma"],x,scalars=[n]);compare(index+1,x)
        torch.cuda.synchronize();del weights,qw,qb
    final=torch.empty(units,4608 if ink else 4096,device="cuda",dtype=torch.bfloat16)
    if ink:
        merged=torch.empty(units,4096,device="cuda",dtype=torch.bfloat16);adapter=torch.empty_like(merged)
        perception.compile_merge2_aot(width=h,class_token=True).launch(x,merged,scalars=[gh,gw])
        media.compile_patch_projection_aot(width=4096,outputs=4096).launch(merged,tensor("model.visual.proj_in.weight"),adapter,scalars=[units])
        activation.launch(adapter,adapter,scalars=[units])
        media.compile_patch_projection_aot(width=4096,outputs=4608).launch(adapter,tensor("model.visual.proj_out.weight"),final,scalars=[units])
    else:
        lower=torch.empty(gh*gw//4,h*9,device="cuda",dtype=torch.bfloat16)
        c1=torch.empty(gh*gw//4,2*h,device="cuda",dtype=torch.bfloat16);c2=torch.empty(units,4*h,device="cuda",dtype=torch.bfloat16)
        perception.compile_conv3_stride2_lower_aot(channels=h).launch(x,lower,scalars=[gh,gw])
        media.compile_biased_projection_aot(width=h*9,outputs=2*h).launch(lower,tensor(prefix+".vit_downsampler1.weight"),tensor(prefix+".vit_downsampler1.bias"),c1,scalars=[gh*gw//4])
        perception.compile_conv3_stride2_lower_aot(channels=2*h).launch(c1,lower,scalars=[gh//2,gw//2])
        media.compile_biased_projection_aot(width=2*h*9,outputs=4*h).launch(lower,tensor(prefix+".vit_downsampler2.weight"),tensor(prefix+".vit_downsampler2.bias"),c2,scalars=[units])
        media.compile_patch_projection_aot(width=4*h,outputs=4096).launch(c2,tensor("vit_large_projector.weight"),final,scalars=[units])
    compare(depth+1,final)
    assert set(stages)==set(range(depth+2))
    print("PE_OWNER_PASS "+json.dumps(dict(kind=args.kind,boundaries=depth+2,byte_exact=True)),flush=True)


if __name__=="__main__":
    with torch.inference_mode():main()
