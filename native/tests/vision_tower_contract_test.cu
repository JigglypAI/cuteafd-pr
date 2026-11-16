// CPU-only descriptor/scratch checks; nvcc compiles both architectures but no
// device is queried, no kernel runs and no backing allocation is made.
#include "../shared/cuda/vision_runtime.cu"
#include <array>

int main() {
  cuteafd_vision_spec s{};
  cuteafd_vision_tower t{};
  cuteafd_vision_block b{};
  std::array<uint64_t,1> missing={UINT64_MAX};
  s.storage_abi=5;s.abi_version=4;s.tower=&t;s.max_tokens=4;s.weight_bytes=1<<20;
  s.hidden=1024;s.intermediate=4096;s.depth=1;s.heads=12;s.kv_heads=12;s.head_dim=128;
  s.patch_size=14;s.output_width=7168;s.blocks=&b;
  s.norm1_bias=missing.data();s.norm2_bias=missing.data();s.q_norm=missing.data();s.k_norm=missing.data();
  s.patch_bias=UINT64_MAX;
  t.version=1;t.size=sizeof(t);t.block_norm_kind=2;t.activation=2;t.block_norm_epsilon=0x1p-23f;
  t.attention_width=1536;t.temporal_patch=1;t.pixel_layout=1;t.rotary_kind=2;
  t.rotary_axis_dims[1]=64;t.rotary_axis_dims[2]=64;t.rotated_dims=128;
  t.inv_freq[0]=UINT64_MAX;t.inv_freq[1]=0;t.inv_freq[2]=128;
  t.absolute_position=UINT64_MAX;t.spatial_merge=2;t.projection_order=1;
  for(auto* norm:{&t.pre_norm,&t.final_norm,&t.post_norm}) {norm->weight=UINT64_MAX;norm->bias=UINT64_MAX;}
  auto& p=t.projectors[0];p.kind=1;p.activation=1;p.input_width=4096;p.hidden_width=4096;p.output_width=7168;
  p.fc1_bias=UINT64_MAX;p.fc2_bias=UINT64_MAX;
  auto& absent=t.projectors[1];absent.fc1=UINT64_MAX;absent.fc1_bias=UINT64_MAX;absent.fc2=UINT64_MAX;absent.fc2_bias=UINT64_MAX;
  b.qkv_bias=UINT64_MAX;b.proj_bias=UINT64_MAX;b.gate_up_bias=UINT64_MAX;b.down_bias=UINT64_MAX;b.key0_bias=UINT64_MAX;
  // Extent validation must reject a backing too small before touching CUDA.
  if(valid_generic_tower(s))return 1;
  s.weight_bytes=1ULL<<30;
  if(!valid_generic_tower(s))return 2;
  GenericTowerStorage scratch;
  size_t bytes=scratch_generic_tower(s,nullptr,scratch);
  if(!bytes || scratch.x || scratch.output || bytes%256)return 3;
  t.temporal_input=2;if(valid_generic_tower(s))return 4;t.temporal_input=0;
  t.version=2;if(valid_generic_tower(s))return 5;t.version=1;
  t.block_norm_epsilon=0;if(valid_generic_tower(s))return 6;t.block_norm_epsilon=0x1p-23f;
  t.projectors[0].input_width=1024;if(valid_generic_tower(s))return 7;t.projectors[0].input_width=4096;
  t.inv_freq[1]=s.weight_bytes;if(valid_generic_tower(s))return 8;t.inv_freq[1]=0;
  for(auto policy: {std::pair<uint32_t,size_t>{16384,1509949696ULL}, {4096,377487616ULL}}) {
    s.max_tokens=policy.first;
    GenericTowerStorage sizing;
    size_t planned=scratch_generic_tower(s,nullptr,sizing);
    if(planned!=policy.second)return 9;
    // Host backing only: exact offsets verify simultaneous operands remain
    // disjoint, while the successive stages share the same physical region.
    std::vector<unsigned char> arena(planned);
    GenericTowerStorage mapped;
    if(scratch_generic_tower(s,arena.data(),mapped)!=planned)return 10;
    size_t n=size_t(s.max_tokens)*4,h=s.hidden,a=t.attention_width;
    auto end=[](const void* pointer,size_t bytes) {return reinterpret_cast<uintptr_t>(pointer)+bytes;};
    if(end(mapped.qkv,n*3*a*2)>reinterpret_cast<uintptr_t>(mapped.q) ||
       end(mapped.q,n*a*2)>reinterpret_cast<uintptr_t>(mapped.k) ||
       end(mapped.k,n*a*2)>reinterpret_cast<uintptr_t>(mapped.v))return 11;
    if(mapped.attention!=mapped.qkv || mapped.norm!=mapped.qkv || mapped.projection!=mapped.next)return 12;
    if(end(mapped.qkv,n*3*a*2)>reinterpret_cast<uintptr_t>(mapped.qkv_norm))return 13;
    if(end(mapped.norm,n*h*2)>reinterpret_cast<uintptr_t>(mapped.mlp))return 14;
    if(end(mapped.output,size_t(s.max_tokens)*s.output_width*2)>reinterpret_cast<uintptr_t>(arena.data()+planned))return 15;
  }
  std::printf("generic-tower-contract-pass scratch=%zu no-device-allocation\n",bytes);
  return 0;
}
