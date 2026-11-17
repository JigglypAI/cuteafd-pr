"""Declarative exports for the independent still-tower ABI (no family loading)."""


def geometry(kind):
    if kind == "kimi":
        return dict(hidden=1024, attention=1536, heads=12, dim=128, intermediate=4096,
                    patch=588, output=7168, temporal=1, group_major=False)
    if kind == "minimax":
        return dict(hidden=1280, attention=1280, heads=16, dim=80, intermediate=5120,
                    patch=1176, output=6144, temporal=2, group_major=True)
    raise ValueError("unknown generic still tower")


def names(kind):
    g=geometry(kind);h,a,i,o=g["hidden"],g["attention"],g["intermediate"],g["output"]
    suffix="raster_t1" if kind=="kimi" else "group_t2"
    result=[f"dynamic_attention_h{g['heads']}_d{g['dim']}",f"patches_{suffix}",f"tables_{suffix}",f"rotary_{suffix}",
            f"residual_h{h}"]
    if kind=="kimi":
        result += ["tower_rms_norm_h1024","tower_rms_norm_h7168","position_h1024","group_h1024"]
        projections=[(588,h),(h,3*a),(a,h),(h,i),(i,h),(4096,4096),(4096,o)]
        result += [f"projection_h{w}_o{n}" for w,n in projections]
        result += ["tanh_gelu_h4096","gelu_h4096"]
    else:
        result += ["tower_layer_norm_h1280"]
        result += [f"biased_projection_h{w}_o{n}" for w,n in
            [(h,3*a),(a,h),(h,i),(i,h),(h,6144),(6144,6144),(24576,6144)]]
        result += [f"projection_h1176_o{h}","gelu_h5120","gelu_h6144"]
    return ["vision_"+name for name in dict.fromkeys(result)]


def builders(kind):
    # Inventory queries stay CPU-only; actual builders verify the pinned source.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
    import _pinned_sparkinfer  # noqa: F401
    import torch
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    from cutlass import Int32,Float32
    from b12x.integration.cuteafd import media,pointwise
    from b12x.integration.cuteafd._common import Operand,Scalar,compile_program
    from b12x.integration.cuteafd.vision_dynamic_attention import compile_dynamic_vision_attention_aot
    from b12x.integration.cuteafd.vision_still import compile_still_patches_aot,compile_spatial_group_aot
    from b12x.integration.cuteafd.vision_rotary_tables import compile_vision_rotary_tables_aot
    from b12x.integration.cuteafd.vision_rotary import compile_vision_qkv_rotary_aot
    from b12x.integration.cuteafd.vision_tower import compile_bilinear_position_aot
    from b12x.integration.cuteafd.vision_gelu import compile_tanh_gelu_aot
    from b12x.integration.cuteafd.vision_rms import VectorizedRms
    from b12x.integration.cuteafd.vision_layer_norm import compile_vectorized_vision_layer_norm_aot

    class RmsFourPointer:
        def __init__(self,width):self.launch=VectorizedRms(width)
        @cute.jit
        def __call__(self,x:cute.Pointer,gamma:cute.Pointer,bias:cute.Pointer,out:cute.Pointer,
                     rows:Int32,epsilon:Float32,stream:cuda.CUstream):
            self.launch(x,gamma,out,rows,epsilon,stream)

    class ExistingRmsFourPointer:
        def __init__(self,width):
            self.launch=pointwise.RuntimeAffineNorm(width,1,1.0,False,False,False,0)
        @cute.jit
        def __call__(self,x:cute.Pointer,gamma:cute.Pointer,bias:cute.Pointer,out:cute.Pointer,
                     rows:Int32,epsilon:Float32,stream:cuda.CUstream):
            self.launch(x,gamma,bias,out,out,out,out,rows,epsilon,stream)

    def rms(width):
        # The1024 route retains its already-qualified striped fold;7168 uses
        # ATen's qualified four-warp vec4 tree. ABI adapter changes no math.
        launch=ExistingRmsFourPointer(width) if width==1024 else RmsFourPointer(width)
        return compile_program(launch,name="vision_tower_rms_adapter",operands=(
            Operand("x",torch.bfloat16,f"[rows,{width}]"),Operand("gamma",torch.bfloat16,f"[{width}]"),
            Operand("bias",torch.bfloat16,"ignored"),Operand("out",torch.bfloat16,f"[rows,{width}]","out")),
            scalars=(Scalar("rows"),Scalar("epsilon","float32")),key=(width,),geometry=dict(width=width))

    g=geometry(kind);h,a,i,o=g["hidden"],g["attention"],g["intermediate"],g["output"]
    suffix="raster_t1" if kind=="kimi" else "group_t2"
    axes=(0,64,64) if kind=="kimi" else (26,26,26)
    result={f"dynamic_attention_h{g['heads']}_d{g['dim']}":lambda:compile_dynamic_vision_attention_aot(heads=g["heads"],dim=g["dim"]),
        f"patches_{suffix}":lambda:compile_still_patches_aot(temporal=g["temporal"],group_major=g["group_major"]),
        f"tables_{suffix}":lambda:compile_vision_rotary_tables_aot(axes=axes,adjacent=kind=="kimi",group_major=g["group_major"]),
        f"rotary_{suffix}":lambda:compile_vision_qkv_rotary_aot(heads=g["heads"],dim=g["dim"],rotated=sum(axes),adjacent=kind=="kimi"),
        f"residual_h{h}":lambda:media.compile_bf16_residual_add_aot(width=h)}
    if kind=="kimi":
        result.update({"tower_rms_norm_h1024":lambda:rms(1024),"tower_rms_norm_h7168":lambda:rms(7168),
            "position_h1024":lambda:compile_bilinear_position_aot(channels=1024),
            "group_h1024":lambda:compile_spatial_group_aot(channels=1024)})
        for w,n in [(588,h),(h,3*a),(a,h),(h,i),(i,h),(4096,4096),(4096,o)]:
            build=media.compile_patch_projection_aot if w%64 else pointwise.compile_bf16_projection_aot
            result[f"projection_h{w}_o{n}"]=lambda w=w,n=n,build=build:build(width=w,outputs=n)
        result["tanh_gelu_h4096"]=lambda:compile_tanh_gelu_aot(width=4096)
        result["gelu_h4096"]=lambda:media.compile_gelu_aot(width=4096)
    else:
        result["tower_layer_norm_h1280"]=lambda:compile_vectorized_vision_layer_norm_aot(width=1280)
        for w,n in [(h,3*a),(a,h),(h,i),(i,h),(h,6144),(6144,6144),(24576,6144)]:
            result[f"biased_projection_h{w}_o{n}"]=lambda w=w,n=n:media.compile_biased_projection_aot(width=w,outputs=n)
        result[f"projection_h1176_o{h}"]=lambda:media.compile_patch_projection_aot(width=1176,outputs=h)
        for width in [5120,6144]:result[f"gelu_h{width}"]=lambda width=width:media.compile_gelu_aot(width=width)
    if ["vision_"+name for name in result]!=names(kind):
        raise ValueError("generic tower export inventory mismatch")
    return result
