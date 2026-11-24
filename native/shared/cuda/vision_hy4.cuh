// Hy4-flash bundled ViT (generic tower, absolute_kind 2 / projection_order 3).
// Serving arithmetic follows Tencent's SGLang tower (sglang@ccb8d9cc
// models/hunyuan_v4_vl.py): BF16 biased patch GEMM, ATen scale-factor bilinear
// position grid added once in BF16, pre-LN blocks with separate biased Q/K/V,
// 16x72 full attention per image, exact GELU MLP, single-rounded RMSNorm, 2x2
// conv merger (channel-major group + GEMM), exact GELU, 1x1 conv, and the
// checkpoint newline row after every merged row. Several images share one
// launch sequence: token-wise programs run over the packed rows and attention
// uses one varlen segment per image. Every launch resolves to a fork AOT program.
#pragma once
constexpr int HY4_MAX_IMAGES=8;
constexpr int HY4_PATCH_WIDTH=3*16*16;

// Merged LM rows of a batch: sum of h/2*(w/2+1). Bounded by units + 8*64.
inline size_t hy4_output_capacity(size_t units) { return units+std::min<size_t>(units,size_t(HY4_MAX_IMAGES)*64); }

size_t scratch_hy4_tower(const cuteafd_vision_spec& s,unsigned char* arena,GenericTowerStorage& out) {
  const size_t units=s.max_tokens,n=units*4,h=s.hidden,i=s.intermediate,o=s.output_width,m=s.tower->projectors[0].hidden_width;
  size_t cursor=0;
  auto take=[&](auto& pointer,size_t bytes) {
    using P=typename std::remove_reference<decltype(pointer)>::type;
    pointer=arena?reinterpret_cast<P>(arena+cursor):nullptr;cursor+=aligned(bytes);
  };
  // Persistent residual stream, block output and varlen segment table.
  take(out.x,n*h*2);take(out.next,n*h*2);take(out.lengths,(HY4_MAX_IMAGES+1)*4);
  const size_t phase_base=cursor;size_t peak=0;
  auto phase=[&]() { peak=std::max(peak,cursor-phase_base);cursor=phase_base; };
  take(out.rgb,n*16*16*3);take(out.lut,768*4);take(out.patches,n*HY4_PATCH_WIDTH*2);phase();
  take(out.position,n*h*2);phase();
  take(out.norm,n*h*2);take(out.q,n*h*2);take(out.k,n*h*2);take(out.v,n*h*2);
  take(out.attention,n*h*2);take(out.lse,n*s.heads*4);phase();
  take(out.norm,n*h*2);take(out.mlp,n*i*2);phase();
  take(out.grouped,units*4*h*2);take(out.projector,units*m*2);take(out.projector_result,units*o*2);
  take(out.output,hy4_output_capacity(units)*o*2);phase();
  return phase_base+peak;
}

#ifdef CUTEAFD_HAVE_VISION_ATTENTION_AOT
// ATen upsample_bilinear2d with scale_factor: the source index uses 1/scale
// computed in double and narrowed to float, exactly as F.interpolate passes it.
inline uint64_t hy4_inverse_scale(int size,int edge) {
  const float value=float(1.0/((double(size)+0.1)/double(edge)));
  uint32_t bits;std::memcpy(&bits,&value,4);return bits;
}
#endif

int encode_hy4_batch(Owner* o,const uint8_t* rgb,uint64_t rgb_bytes,const float* lut,const int32_t* grids,int images,
    uint16_t* output,uint64_t output_bytes,cuteafd_vision_observer cb,void* ctx) {
#ifndef CUTEAFD_HAVE_VISION_ATTENTION_AOT
  return cudaErrorNotSupported;
#else
  const auto& s=o->spec;const auto& t=*s.tower;auto& b=o->generic_storage;
  if(!o->uploaded || !rgb || !lut || !grids || !output || images<=0 || images>HY4_MAX_IMAGES || t.projection_order!=3)
    return cudaErrorInvalidValue;
  int64_t patches=0,units=0,rows=0,bytes=0;
  int32_t lengths[HY4_MAX_IMAGES+1]={0};
  for(int image=0;image<images;++image) {
    const int gh=grids[2*image],gw=grids[2*image+1];
    if(gh<2 || gw<2 || gh%2 || gw%2 || gh>int(t.position_height) || gw>int(t.position_width))return cudaErrorInvalidValue;
    patches+=int64_t(gh)*gw;units+=int64_t(gh/2)*(gw/2);rows+=int64_t(gh/2)*(gw/2+1);
    bytes+=int64_t(gh)*gw*16*16*3;lengths[image+1]=int32_t(patches);
  }
  for(int image=images;image<HY4_MAX_IMAGES;++image)lengths[image+1]=int32_t(patches);
  if(patches>int64_t(s.max_tokens)*4 || units>int64_t(s.max_tokens) || uint64_t(bytes)!=rgb_bytes ||
     output_bytes!=uint64_t(rows)*s.output_width*2 || uint64_t(rows)>hy4_output_capacity(s.max_tokens))return cudaErrorInvalidValue;
  int e=device_ok(o);if(e)return e;
  const int n=int(patches),h=int(s.hidden),out=int(s.output_width),inter=int(s.intermediate);
  const int hidden=int(t.projectors[0].hidden_width);
  auto stream=o->stream;
  auto at=[&](uint64_t offset){return static_cast<void*>(o->weights+offset);};
  const std::string H=std::to_string(h);
  auto biased=[&](uint16_t* x,uint64_t w,uint64_t bias,uint16_t* y,int count,int in,int width) {
    return generic_program(o,"vision_biased_projection_h"+std::to_string(in)+"_o"+std::to_string(width),
      {x,at(w),at(bias),y},{uint64_t(count)});
  };
#define HY4_RUN(expr) do { int rc=int(expr);if(rc)return drain(o,rc); } while(0)
  // One upload of every image and one download of every span, one sync.
  HY4_RUN(cudaMemcpyAsync(b.rgb,rgb,rgb_bytes,cudaMemcpyHostToDevice,stream));
  HY4_RUN(cudaMemcpyAsync(b.lut,lut,768*4,cudaMemcpyHostToDevice,stream));
  HY4_RUN(cudaMemcpyAsync(b.lengths,lengths,sizeof(lengths),cudaMemcpyHostToDevice,stream));
  {
    int64_t patch=0,pixel=0;
    for(int image=0;image<images;++image) {
      const int gh=grids[2*image],gw=grids[2*image+1];
      HY4_RUN(generic_program(o,"vision_patches_raster_t1",{b.rgb+pixel,b.lut,b.patches+patch*HY4_PATCH_WIDTH},
        {uint64_t(gh*16),uint64_t(gw*16)}));
      patch+=int64_t(gh)*gw;pixel+=int64_t(gh)*gw*16*16*3;
    }
  }
  HY4_RUN(biased(b.patches,s.patch,s.patch_bias,b.x,n,HY4_PATCH_WIDTH,h));
  HY4_RUN(observe(o,cb,ctx,0,b.x,n,h,false));
  {
    // Row 0 of the learned table belongs to the unused class token.
    const uint64_t table=t.absolute_position+uint64_t(h)*2;
    int64_t patch=0;
    for(int image=0;image<images;++image) {
      const int gh=grids[2*image],gw=grids[2*image+1];
      HY4_RUN(generic_program(o,"vision_scaled_position_h"+H,{at(table),b.position+patch*h},
        {uint64_t(gh),uint64_t(gw),hy4_inverse_scale(gh,int(t.position_height)),hy4_inverse_scale(gw,int(t.position_width))}));
      patch+=int64_t(gh)*gw;
    }
  }
  HY4_RUN(generic_residual(o,b.x,b.position,n));
  HY4_RUN(observe(o,cb,ctx,-1,b.x,n,h,false));
  HY4_RUN(generic_teacher_boundary(o,0,b.x,n,h));
  const float scale=1.f/sqrtf(float(s.head_dim));
  const uint64_t matrix=uint64_t(h)*h*2,vector=uint64_t(h)*2;
  for(uint32_t layer=0;layer<s.depth;++layer) {
    const auto& block=s.blocks[layer];
    cuteafd_vision_tower_norm norm1{1,t.block_norm_epsilon,block.norm1,s.norm1_bias[layer]};
    cuteafd_vision_tower_norm norm2{1,t.block_norm_epsilon,block.norm2,s.norm2_bias[layer]};
    HY4_RUN(generic_norm(o,norm1,b.x,b.norm,n,h));
    uint16_t* qkv[3]={b.q,b.k,b.v};
    for(int part=0;part<3;++part)HY4_RUN(biased(b.norm,block.qkv+part*matrix,block.qkv_bias+part*vector,qkv[part],n,h,h));
    HY4_RUN(cuteafd_vision_dynamic_attention_launch(o->attention,b.q,b.k,b.v,b.attention,b.lse,b.lengths,n,scale,stream));
    HY4_RUN(biased(b.attention,block.proj,block.proj_bias,b.next,n,h,h));
    HY4_RUN(generic_residual(o,b.x,b.next,n));
    HY4_RUN(generic_norm(o,norm2,b.x,b.norm,n,h));
    HY4_RUN(biased(b.norm,block.gate_up,block.gate_up_bias,b.mlp,n,h,inter));
    HY4_RUN(generic_activation(o,b.mlp,n,inter,t.activation));
    HY4_RUN(biased(b.mlp,block.down,block.down_bias,b.next,n,inter,h));
    HY4_RUN(generic_residual(o,b.x,b.next,n));
    HY4_RUN(observe(o,cb,ctx,int(layer)+1,b.x,n,h,false));
    if(layer+1<s.depth)HY4_RUN(generic_teacher_boundary(o,int(layer)+1,b.x,n,h));
  }
  HY4_RUN(generic_norm(o,t.final_norm,b.x,b.next,n,h));
  HY4_RUN(observe(o,cb,ctx,int(s.depth)+1,b.next,n,h,false));
  {
    int64_t patch=0,unit=0;
    for(int image=0;image<images;++image) {
      const int gh=grids[2*image],gw=grids[2*image+1];
      HY4_RUN(generic_program(o,"vision_group_channel_h"+H,{b.next+patch*h,b.grouped+unit*4*h},
        {uint64_t(gh),uint64_t(gw)}));
      patch+=int64_t(gh)*gw;unit+=int64_t(gh/2)*(gw/2);
    }
  }
  const auto& projector=t.projectors[0];
  HY4_RUN(biased(b.grouped,projector.fc1,projector.fc1_bias,b.projector,int(units),4*h,hidden));
  HY4_RUN(generic_activation(o,b.projector,int(units),hidden,projector.activation));
  HY4_RUN(biased(b.projector,projector.fc2,projector.fc2_bias,b.projector_result,int(units),hidden,out));
  {
    int64_t unit=0,row=0;
    for(int image=0;image<images;++image) {
      const int gh=grids[2*image],gw=grids[2*image+1];
      HY4_RUN(generic_program(o,"vision_row_markers_h"+std::to_string(out),
        {b.projector_result+unit*out,at(s.merger_extra[0]),b.output+row*out},{uint64_t(gh/2),uint64_t(gw/2)}));
      unit+=int64_t(gh/2)*(gw/2);row+=int64_t(gh/2)*(gw/2+1);
    }
  }
  HY4_RUN(observe(o,cb,ctx,int(s.depth)+2,b.output,int(rows),out,false));
  HY4_RUN(cudaMemcpyAsync(output,b.output,output_bytes,cudaMemcpyDeviceToHost,stream));
  HY4_RUN(cudaStreamSynchronize(stream));o->ledger.encodes+=uint64_t(images);
#undef HY4_RUN
  return 0;
#endif
}
