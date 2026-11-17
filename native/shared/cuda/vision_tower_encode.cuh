// Generic still encoder sequence. All storage is admitted before registration;
// every arithmetic launch resolves to a fork AOT program owned by attention.
#ifdef CUTEAFD_HAVE_VISION_ATTENTION_AOT
int generic_activation(Owner* o,uint16_t* x,int rows,int width,uint32_t activation) {
  const std::string name="vision_"+std::string(activation==1?"gelu":"tanh_gelu")+"_h"+std::to_string(width);
  return generic_program(o,name,{x,x},{uint64_t(rows)});
}
int generic_residual(Owner* o,uint16_t* x,uint16_t* addition,int rows) {
  return generic_program(o,"vision_residual_h"+std::to_string(o->spec.hidden),{x,addition,x},{uint64_t(rows)});
}
int generic_project(Owner* o,const cuteafd_vision_tower_projector& p,uint16_t* input,
    uint16_t* hidden,uint16_t* output,int rows) {
  int e=generic_projection(o,input,p.fc1,p.fc1_bias,hidden,rows,p.input_width,p.hidden_width);if(e)return e;
  e=generic_activation(o,hidden,rows,p.hidden_width,p.activation);if(e)return e;
  return generic_projection(o,hidden,p.fc2,p.fc2_bias,output,rows,p.hidden_width,p.output_width);
}
#endif

int encode_generic_tower(Owner* o,const uint8_t* rgb,uint64_t rgb_bytes,const float* lut,int gh,int gw,
    int source_h,int source_w,int cx,int cy,int cw,int ch,uint16_t* output,uint64_t output_bytes,
    cuteafd_vision_observer cb,void* ctx) {
#ifndef CUTEAFD_HAVE_VISION_ATTENTION_AOT
  return cudaErrorNotSupported;
#else
  const auto& s=o->spec;const auto& t=*s.tower;auto& b=o->generic_storage;
  const int64_t count=int64_t(gh)*gw;
  if(!o->uploaded || !rgb || !lut || !output || gh<=0 || gw<=0 || gh%2 || gw%2 ||
     count>int64_t(s.max_tokens)*4 || source_h!=gh*int(s.patch_size) || source_w!=gw*int(s.patch_size) ||
     cx<0 || cy<0 || cw<=0 || ch<=0 || int64_t(cx)+cw>source_w || int64_t(cy)+ch>source_h ||
     rgb_bytes!=uint64_t(source_h)*source_w*3 || output_bytes!=uint64_t(count/4)*s.output_width*2)return cudaErrorInvalidValue;
  int e=device_ok(o);if(e)return e;
  const int n=int(count),units=n/4,h=int(s.hidden),a=int(t.attention_width);
  auto stream=o->stream;
#define GENERIC_RUN(expr) do { int rc=int(expr);if(rc)return drain(o,rc); } while(0)
  GENERIC_RUN(cudaMemcpyAsync(b.rgb,rgb,rgb_bytes,cudaMemcpyHostToDevice,stream));
  GENERIC_RUN(cudaMemcpyAsync(b.lut,lut,768*4,cudaMemcpyHostToDevice,stream));
  std::string suffix=t.pixel_layout==2?"group_t2":"raster_t1";
  GENERIC_RUN(generic_program(o,"vision_patches_"+suffix,{b.rgb,b.lut,b.patches},{uint64_t(source_h),uint64_t(source_w)}));
  GENERIC_RUN(generic_projection(o,b.patches,s.patch,s.patch_bias,b.x,n,3*t.temporal_patch*s.patch_size*s.patch_size,h));
  GENERIC_RUN(observe(o,cb,ctx,0,b.x,n,h,false));
  if(t.absolute_kind) {
    GENERIC_RUN(generic_program(o,"vision_position_h"+std::to_string(h),
      {o->weights+t.absolute_position,b.position},{uint64_t(gh),uint64_t(gw)}));
    GENERIC_RUN(generic_residual(o,b.x,b.position,n));
    GENERIC_RUN(observe(o,cb,ctx,-1,b.x,n,h,false));
  }
  GENERIC_RUN(generic_norm(o,t.pre_norm,b.x,b.next,n,h));
  std::swap(b.x,b.next);
  const auto frequency=[&](int axis)->void* {
    return t.inv_freq[axis]==UINT64_MAX?static_cast<void*>(b.cosines):o->weights+t.inv_freq[axis];
  };
  GENERIC_RUN(generic_program(o,"vision_tables_"+suffix,
    {frequency(0),frequency(1),frequency(2),b.cosines,b.sines},{uint64_t(gh),uint64_t(gw)}));
  int32_t lengths[2]={0,n};
  GENERIC_RUN(cudaMemcpyAsync(b.lengths,lengths,sizeof(lengths),cudaMemcpyHostToDevice,stream));
  int l2=0;GENERIC_RUN(cudaDeviceGetAttribute(&l2,cudaDevAttrL2CacheSize,o->device));
  const float scale=1.f/sqrtf(float(s.head_dim));uint32_t scale_bits;std::memcpy(&scale_bits,&scale,4);
  for(uint32_t layer=0;layer<s.depth;++layer) {
    const auto& block=s.blocks[layer];
    cuteafd_vision_tower_norm norm1{t.block_norm_kind,t.block_norm_epsilon,block.norm1,s.norm1_bias[layer]};
    cuteafd_vision_tower_norm norm2{t.block_norm_kind,t.block_norm_epsilon,block.norm2,s.norm2_bias[layer]};
    GENERIC_RUN(generic_norm(o,norm1,b.x,b.qkv_norm,n,h));
    GENERIC_RUN(generic_projection(o,b.qkv_norm,block.qkv,block.qkv_bias,b.qkv,n,h,3*a));
    GENERIC_RUN(generic_program(o,"vision_rotary_"+suffix,
      {b.qkv,b.cosines,b.sines,b.q,b.k,b.v},{uint64_t(n)}));
    GENERIC_RUN(generic_program(o,"vision_dynamic_attention_h"+std::to_string(s.heads)+"_d"+std::to_string(s.head_dim),
      {b.q,b.k,b.v,b.attention,b.lse,b.lengths},{uint64_t(n),scale_bits,uint64_t(l2)}));
    GENERIC_RUN(generic_projection(o,b.attention,block.proj,block.proj_bias,b.next,n,a,h));
    GENERIC_RUN(generic_residual(o,b.x,b.next,n));
    GENERIC_RUN(generic_norm(o,norm2,b.x,b.norm,n,h));
    GENERIC_RUN(generic_projection(o,b.norm,block.gate_up,block.gate_up_bias,b.mlp,n,h,s.intermediate));
    GENERIC_RUN(generic_activation(o,b.mlp,n,s.intermediate,t.activation));
    GENERIC_RUN(generic_projection(o,b.mlp,block.down,block.down_bias,b.next,n,s.intermediate,h));
    GENERIC_RUN(generic_residual(o,b.x,b.next,n));
    GENERIC_RUN(observe(o,cb,ctx,int(layer)+1,b.x,n,h,false));
  }
  GENERIC_RUN(generic_norm(o,t.final_norm,b.x,b.next,n,h));
  if(t.projection_order==2) {
    GENERIC_RUN(generic_project(o,t.projectors[0],b.next,b.preprojector,b.preprojected,n));
    // MiniMax source rows already group the2x2 neighbors contiguously. Reuse
    // that exact view; copying through a phase alias would overwrite it.
  } else {
    GENERIC_RUN(generic_program(o,"vision_group_h"+std::to_string(h),{b.next,b.grouped},{uint64_t(gh),uint64_t(gw)}));
  }
  const auto& projector=t.projectors[t.projection_order==2?1:0];
  GENERIC_RUN(generic_project(o,projector,b.grouped,b.projector,b.projector_result,units));
  GENERIC_RUN(observe(o,cb,ctx,int(s.depth)+1,b.projector_result,units,s.output_width,false));
  GENERIC_RUN(generic_norm(o,t.post_norm,b.postnorm_input,b.output,units,s.output_width));
  GENERIC_RUN(observe(o,cb,ctx,int(s.depth)+2,b.output,units,s.output_width,false));
  GENERIC_RUN(cudaMemcpyAsync(output,b.output,output_bytes,cudaMemcpyDeviceToHost,stream));
  GENERIC_RUN(cudaStreamSynchronize(stream));++o->ledger.encodes;
#undef GENERIC_RUN
  return 0;
#endif
}
