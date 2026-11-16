// Host-only benchmark of the production launch wrapper. CUDA submission is a
// no-op, isolating registry/counter cost; this does not measure the CUDA driver.
// Compile with function/data sections and --gc-sections, no CUDA linkage.
#ifndef CUTEAFD_BENCH_SOURCE
#define CUTEAFD_BENCH_SOURCE "cuteafd_native.cc"
#endif
#include CUTEAFD_BENCH_SOURCE
#include <thread>

extern "C" const char* cudaGetErrorString(cudaError_t) { return "stub"; }
extern "C" cudaError_t cudaGetDevice(int* device) { *device = 0; return cudaSuccess; }
extern "C" cudaError_t cudaSetDevice(int) { return cudaSuccess; }
extern "C" cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }
extern "C" cudaError_t cudaGetLastError() { return cudaSuccess; }
extern "C" cudaError_t cudaGraphExecDestroy(cudaGraphExec_t) { return cudaSuccess; }

extern "C" __attribute__((noinline)) cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t) {
  asm volatile("" ::: "memory");
  return cudaSuccess;
}

int main() {
#ifdef CUTEAFD_BENCH_EXECUTOR
  void* checked;
  {
    std::lock_guard<std::mutex> guard(graph_handles_mutex);
    checked = own_graph_exec(reinterpret_cast<cudaGraphExec_t>(uintptr_t(2)));
  }
  std::thread wrong_thread([&] {
    if (cuteafd_cuda_graph_launch(checked, reinterpret_cast<void*>(uintptr_t(2))) == CUTEAFD_STATUS_OK) std::abort();
    if (cuteafd_cuda_graph_exec_destroy(checked) == CUTEAFD_STATUS_OK) std::abort();
  });
  wrong_thread.join();
  if (cuteafd_cuda_graph_launch(checked, reinterpret_cast<void*>(uintptr_t(2))) != CUTEAFD_STATUS_OK) std::abort();
  if (cuteafd_cuda_graph_exec_destroy(checked) != CUTEAFD_STATUS_OK) std::abort();
#endif
  constexpr size_t iterations = 1000000;
  for (unsigned workers : {1u, 2u}) {
    std::atomic<unsigned> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> threads;
    for (unsigned worker = 0; worker < workers; ++worker) {
      threads.emplace_back([&, worker] {
        void* handle = reinterpret_cast<void*>(uintptr_t(2 + worker * 2));
#ifndef CUTEAFD_BENCH_RAW
        {
          std::lock_guard<std::mutex> guard(graph_handles_mutex);
          handle = own_graph_exec(reinterpret_cast<cudaGraphExec_t>(handle));
        }
#endif
        ready.fetch_add(1);
        while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
        const auto start = std::chrono::steady_clock::now();
        auto volatile launch = &cuteafd_cuda_graph_launch;
        for (size_t i = 0; i < iterations; ++i) {
          if (launch(handle, reinterpret_cast<void*>(uintptr_t(2))) != CUTEAFD_STATUS_OK)
            std::abort();
        }
        const auto ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
        std::printf("native_launch workers=%u worker=%u ns=%.2f iterations=%zu\n", workers, worker, ns / iterations, iterations);
      });
    }
    while (ready.load() != workers) std::this_thread::yield();
    go.store(true, std::memory_order_release);
    for (auto& thread : threads) thread.join();
  }
}
