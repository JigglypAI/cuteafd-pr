// Test-only long kernel: globaltimer is nanoseconds and independent of SM clock.
#include <cuda_runtime.h>
#include <cstdint>

__global__ void graph_update_spin(uint64_t nanoseconds, uint32_t* output,
    uint32_t value, uint64_t* times) {
  uint64_t begin, now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(begin));
  do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now)); }
  while (now - begin < nanoseconds);
  *output = value;
  times[0] = begin;
  times[1] = now;
}

extern "C" const void* cuteafd_test_graph_spin_kernel() {
  return reinterpret_cast<const void*>(graph_update_spin);
}
