// Shared PE arithmetic. Every live image extent is launch data; storage belongs
// to the admitted resident owner, and all arithmetic comes from reviewed AOT.
bool valid_perception(const cuteafd_vision_spec& s) {
  if(s.abi_version!=3 || !s.perception || s.tower || (s.reserved!=6 && s.reserved!=7) ||
     !s.max_tokens || s.max_tokens>4096 || (s.reserved==7 && s.max_tokens<81) || s.geometry_reserved || s.norm_eps!=1e-5f)return false;
  bool ink=s.reserved==6;const auto& p=*s.perception;
  uint32_t h=ink?1024:1536,i=ink?4096:8960,d=ink?23:47;
  if(s.hidden!=h || s.depth!=d || s.heads!=16 || s.kv_heads!=16 || s.head_dim!=h/16 ||
     s.intermediate!=i || s.patch_size!=(ink?20:14) || s.output_width!=(ink?4608:4096) ||
     s.merger_width!=h*4 || p.activation!=uint32_t(!ink) || p.position_grid!=(ink?32:52) ||
     p.class_token!=uint32_t(ink) || p.output_kind!=uint32_t(!ink) || !p.scale1 || !p.scale2 ||
     !s.weight_bytes || s.weight_bytes>(5ULL<<30))return false;
  const uint64_t offsets[]={s.patch,s.pos_embed,s.inv_freq,p.ln_pre,p.ln_pre_bias};
  const size_t bytes[]={size_t(h)*3*s.patch_size*s.patch_size*2,size_t(p.position_grid)*p.position_grid*h*2,
    size_t(s.head_dim/4)*4,size_t(h)*2,size_t(h)*2};
  for(int j=0;j<5;++j)if(!extent(s,offsets[j],bytes[j]))return false;
  if(ink) {
    if(!extent(s,p.class_embedding,h*2) || !extent(s,p.class_position,h*2) ||
       !extent(s,s.merger_fc1,size_t(4096)*4096*2) || !extent(s,s.merger_fc2,size_t(s.output_width)*4096*2))return false;
  } else {
    if(!extent(s,p.downsample1,size_t(3072)*1536*9*2) || !extent(s,p.downsample1_bias,3072*2) ||
       !extent(s,p.downsample2,size_t(6144)*3072*9*2) || !extent(s,p.downsample2_bias,6144*2) ||
       !extent(s,s.merger_fc2,size_t(4096)*6144*2))return false;
  }
  for(uint32_t layer=0;layer<d;++layer) {
    const auto& b=s.blocks[layer];
    const uint64_t off[]={b.qkv,b.qkv_bias,b.proj,b.proj_bias,b.gate_up,b.gate_up_bias,b.down,b.down_bias,
      b.norm1,b.norm2,s.norm1_bias[layer],s.norm2_bias[layer],p.scale1[layer],p.scale2[layer]};
    const size_t len[]={size_t(3*h)*h*2,3*h*2,size_t(h)*h*2,h*2,size_t(i)*h*2,i*2,size_t(h)*i*2,h*2,
      h*2,h*2,h*2,h*2,h*2,h*2};
    for(int j=0;j<14;++j)if(!extent(s,off[j],len[j]))return false;
    if(b.window || b.column_order || b.key0_bias!=UINT64_MAX || s.q_norm[layer]!=UINT64_MAX || s.k_norm[layer]!=UINT64_MAX)return false;
  }
  return true;
}

size_t scratch_perception(Owner* o) {
  const auto& s=o->spec;bool ink=s.reserved==6;
  // Step accepts the official global52 or local36 grids only. Inkling's
  // declared merged-row capacity also includes one live CLS attention row.
  size_t n=ink?std::min(size_t(s.max_tokens)*4+1,size_t(16384)):
    (s.max_tokens>=169?size_t(2704):size_t(1296));
  size_t cursor=0;
  auto take=[&](auto& ptr,size_t bytes) {
    using P=typename std::remove_reference<decltype(ptr)>::type;
    ptr=o->arena?reinterpret_cast<P>(o->arena+cursor):nullptr;cursor+=aligned(bytes);
  };
  take(o->muse_x,n*s.hidden*2);take(o->muse_xt,n*s.hidden*2);take(o->norm,n*s.hidden*2);
  take(o->patches,n*3*s.patch_size*s.patch_size*2);take(o->muse_qkv,n*3*s.hidden*2);
  take(o->q,16384*s.hidden*2);take(o->k,16384*s.hidden*2);take(o->v,16384*s.hidden*2);take(o->attn,16384*s.hidden*2);
  take(o->muse_proj,n*s.hidden*2);take(o->mlp,n*s.intermediate*2);
  take(o->muse_adapter,size_t(s.max_tokens)*4096*2);take(o->output,size_t(s.max_tokens)*s.output_width*2);
  take(o->pe_merged,size_t(s.max_tokens)*s.merger_width*2);
  take(o->cu_seqlens,2*4);take(o->lse,16384*16*4);
  take(o->lut,768*4);
  size_t source=ink?n*20*20*3:size_t(3024)*3024*3;
  take(o->rgb,source);
  if(!ink) {
    take(o->pe_pixels,size_t(3024)*3024*3*4);
    take(o->pe_horizontal,size_t(3024)*728*3*4);take(o->pe_resized,size_t(728)*728*3*4);
    take(o->pe_ix,728*4);take(o->pe_iy,728*4);take(o->pe_nx,728*4);take(o->pe_ny,728*4);
    take(o->pe_wx,728*64*4);take(o->pe_wy,728*64*4);
    take(o->pe_conv_lower,std::max(n/4*1536*9*2,n/16*3072*9*2));take(o->pe_conv1,n/4*3072*2);take(o->pe_conv2,n/16*6144*2);
  }
  return cursor;
}

#ifdef CUTEAFD_HAVE_VISION_ATTENTION_AOT
int pe_program(Owner* o,const std::string& name,std::initializer_list<void*> pointers,std::initializer_list<uint64_t> scalars) {
  return cuteafd_vision_pointwise_launch(o->attention,name.c_str(),pointers.begin(),scalars.begin(),o->stream);
}
int pe_norm(Owner* o,uint16_t* x,uint64_t w,uint64_t b,uint16_t* out,int rows) {
  uint32_t eps;std::memcpy(&eps,&o->spec.norm_eps,4);
  return pe_program(o,"vision_pe_norm_h"+std::to_string(o->spec.hidden),{x,o->weights+w,o->weights+b,out},{uint64_t(rows),eps});
}
int pe_projection(Owner* o,uint16_t* x,uint64_t w,uint64_t b,uint16_t* out,int rows,int input,int output) {
  bool patch_specialization=b==UINT64_MAX && output==4096 && (input==4096 || input==6144);
  std::string name="vision_"+std::string(patch_specialization?"pe_patch_projection":b==UINT64_MAX?"projection":"biased_projection")+
    "_h"+std::to_string(input)+"_o"+std::to_string(output);
  if(b==UINT64_MAX)return pe_program(o,name,{x,o->weights+w,out},{uint64_t(rows)});
  return pe_program(o,name,{x,o->weights+w,o->weights+b,out},{uint64_t(rows)});
}
#endif

int encode_perception(Owner* o,const uint8_t* rgb,uint64_t rgb_bytes,const float* parameters,
    int gh,int gw,int source_h,int source_w,int cx,int cy,int cw,int ch,
    uint16_t* output,uint64_t output_bytes,cuteafd_vision_observer cb,void* ctx) {
#ifndef CUTEAFD_HAVE_VISION_ATTENTION_AOT
  return cudaErrorNotSupported;
#else
  const auto& s=o->spec;const auto& p=*s.perception;bool ink=s.reserved==6;
  int64_t patches=int64_t(gh)*gw,n=patches+int(ink),units=ink?patches/4:((gh+3)/4)*((gw+3)/4);
  if(!o->uploaded || !rgb || !parameters || !output || gh<=0 || gw<=0 || source_h<=0 || source_w<=0 ||
     gh%2 || gw%2 || n>16384 || units>s.max_tokens || source_h>3024 || source_w>3024 ||
     rgb_bytes!=uint64_t(source_h)*source_w*3 || output_bytes!=uint64_t(units)*s.output_width*2 ||
     cx<0 || cy<0 || cw<=0 || ch<=0 || int64_t(cx)+cw>source_w || int64_t(cy)+ch>source_h ||
     (ink && (source_h!=gh*20 || source_w!=gw*20)) || (!ink && (gh!=gw || (gh!=52 && gh!=36) || cx || cy || cw!=source_w || ch!=source_h)))return cudaErrorInvalidValue;
  int e=device_ok(o);if(e)return e;auto stream=o->stream;
#define PE_RUN(expr) do {int rc=int(expr);if(rc)return drain(o,rc);} while(0)
  PE_RUN(cudaMemcpyAsync(o->rgb,rgb,rgb_bytes,cudaMemcpyHostToDevice,stream));
  PE_RUN(cudaMemcpyAsync(o->lut,parameters,768*4,cudaMemcpyHostToDevice,stream));
  auto* w=o->weights;
  if(ink) {
    PE_RUN(pe_program(o,"vision_pe_rgb_p20",{o->rgb,o->lut,o->lut+3,o->lut+6,o->patches},
      {uint64_t(source_h),uint64_t(source_w),uint64_t(cx),uint64_t(cy),uint64_t(cw),uint64_t(ch)}));
  } else {
    int oh=gh*14,ow=gw*14;
    auto taps=[](int input,int output){float scale=float(input)/float(output);return 2*int(ceilf(fmaxf(1.f,scale)))+1;};
    int tx=taps(source_w,ow),ty=taps(source_h,oh);if(tx>64 || ty>64)return drain(o,cudaErrorInvalidValue);
    PE_RUN(pe_program(o,"vision_rgb_normalize",{o->rgb,o->lut,o->lut+3,o->pe_pixels},{uint64_t(source_h),uint64_t(source_w)}));
    PE_RUN(pe_program(o,"vision_resize_coefficients_bilinear",{o->pe_ix,o->pe_nx,o->pe_wx},{uint64_t(source_w),uint64_t(ow),uint64_t(tx)}));
    PE_RUN(pe_program(o,"vision_resize_coefficients_bilinear",{o->pe_iy,o->pe_ny,o->pe_wy},{uint64_t(source_h),uint64_t(oh),uint64_t(ty)}));
    PE_RUN(pe_program(o,"vision_resize_horizontal",{o->pe_pixels,o->pe_ix,o->pe_nx,o->pe_wx,o->pe_horizontal},
      {uint64_t(source_h),uint64_t(source_w),uint64_t(ow),uint64_t(tx)}));
    PE_RUN(pe_program(o,"vision_resize_vertical",{o->pe_horizontal,o->pe_iy,o->pe_ny,o->pe_wy,o->pe_resized},
      {uint64_t(source_h),uint64_t(ow),uint64_t(oh),uint64_t(ty)}));
    PE_RUN(pe_program(o,"vision_pe_patches_p14",{o->pe_resized,o->patches},{uint64_t(oh),uint64_t(ow)}));
  }
  PE_RUN(pe_projection(o,o->patches,s.patch,UINT64_MAX,o->muse_x,int(patches),3*s.patch_size*s.patch_size,s.hidden));
  auto h=std::to_string(s.hidden);
  uint64_t cls=ink?p.class_embedding:p.ln_pre,cls_pos=ink?p.class_position:p.ln_pre_bias;
  PE_RUN(pe_program(o,"vision_pe_position_h"+h,{o->muse_x,w+s.pos_embed,w+cls,w+cls_pos,o->muse_xt},{uint64_t(gh),uint64_t(gw)}));
  PE_RUN(pe_norm(o,o->muse_xt,p.ln_pre,p.ln_pre_bias,o->muse_x,int(n)));
  PE_RUN(observe(o,cb,ctx,0,o->muse_x,int(n),s.hidden,false));
  auto* x=o->muse_x;auto* xt=o->muse_xt;
  vision_lengths<<<1,1,0,stream>>>(o->cu_seqlens,int(n));PE_RUN(cudaGetLastError());
  for(uint32_t layer=0;layer<s.depth;++layer) {
    const auto& b=s.blocks[layer];
    PE_RUN(pe_norm(o,x,b.norm1,s.norm1_bias[layer],o->norm,int(n)));
    PE_RUN(pe_projection(o,o->norm,b.qkv,b.qkv_bias,o->muse_qkv,int(n),s.hidden,3*s.hidden));
    PE_RUN(pe_program(o,"vision_pe_rotary_d"+std::to_string(s.head_dim),{o->muse_qkv,w+s.inv_freq,o->q,o->k,o->v},{uint64_t(n),uint64_t(gw)}));
    // Vendor computes head_dim**-0.5 in double before the FP32 scalar cast.
    // Float sqrt then reciprocal is one ULP low at Step's D96.
    PE_RUN(cuteafd_vision_attention_launch(o->attention,o->q,o->k,o->v,o->attn,
      o->lse,o->cu_seqlens,float(1.0/std::sqrt(double(s.head_dim))),stream));
    PE_RUN(pe_projection(o,o->attn,b.proj,b.proj_bias,o->muse_proj,int(n),s.hidden,s.hidden));
    PE_RUN(pe_program(o,"vision_pe_scale_h"+h,{x,o->muse_proj,w+p.scale1[layer],xt},{uint64_t(n)}));std::swap(x,xt);
    PE_RUN(pe_norm(o,x,b.norm2,s.norm2_bias[layer],o->norm,int(n)));
    PE_RUN(pe_projection(o,o->norm,b.gate_up,b.gate_up_bias,o->mlp,int(n),s.hidden,s.intermediate));
    PE_RUN(pe_program(o,ink?"vision_gelu_h4096":"vision_pe_quick_gelu_h8960",{o->mlp,o->mlp},{uint64_t(n)}));
    PE_RUN(pe_projection(o,o->mlp,b.down,b.down_bias,o->muse_proj,int(n),s.intermediate,s.hidden));
    PE_RUN(pe_program(o,"vision_pe_scale_h"+h,{x,o->muse_proj,w+p.scale2[layer],xt},{uint64_t(n)}));std::swap(x,xt);
    PE_RUN(observe(o,cb,ctx,int(layer)+1,x,int(n),s.hidden,false));
  }
  if(ink) {
    PE_RUN(pe_program(o,"vision_pe_merge_h1024",{x,o->pe_merged},{uint64_t(gh),uint64_t(gw)}));
    PE_RUN(pe_projection(o,o->pe_merged,s.merger_fc1,UINT64_MAX,o->muse_adapter,int(units),4096,4096));
    PE_RUN(pe_program(o,"vision_gelu_h4096",{o->muse_adapter,o->muse_adapter},{uint64_t(units)}));
    PE_RUN(pe_projection(o,o->muse_adapter,s.merger_fc2,UINT64_MAX,o->output,int(units),4096,s.output_width));
  } else {
    PE_RUN(pe_program(o,"vision_pe_conv_lower_h1536",{x,o->pe_conv_lower},{uint64_t(gh),uint64_t(gw)}));
    PE_RUN(pe_projection(o,o->pe_conv_lower,p.downsample1,p.downsample1_bias,o->pe_conv1,gh*gw/4,1536*9,3072));
    PE_RUN(pe_program(o,"vision_pe_conv_lower_h3072",{o->pe_conv1,o->pe_conv_lower},{uint64_t(gh/2),uint64_t(gw/2)}));
    PE_RUN(pe_projection(o,o->pe_conv_lower,p.downsample2,p.downsample2_bias,o->pe_conv2,int(units),3072*9,6144));
    PE_RUN(pe_projection(o,o->pe_conv2,s.merger_fc2,UINT64_MAX,o->output,int(units),6144,s.output_width));
  }
  PE_RUN(observe(o,cb,ctx,s.depth+1,o->output,int(units),s.output_width,false));
  PE_RUN(cudaMemcpyAsync(output,o->output,output_bytes,cudaMemcpyDeviceToHost,stream));
  PE_RUN(cudaStreamSynchronize(stream));++o->ledger.encodes;
#undef PE_RUN
  return 0;
#endif
}
