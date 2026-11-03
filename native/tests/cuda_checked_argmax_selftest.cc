#include "cuteafd_native.h"
#ifdef CUTEAFD_TEST_GENERIC_AOT
#include "cuteafd_dsv4_programs.h"
#endif
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>
static void check(bool value) { if (!value) std::abort(); }
static void ok(cuteafd_status_t value) { check(value == CUTEAFD_STATUS_OK); }
#ifdef CUTEAFD_TEST_GENERIC_AOT
static void cuda_ok(cudaError_t status) { check(status == cudaSuccess); }

static uint32_t stats_program() {
  for (uint32_t i = 0; i < cuteafd_dsv4_program_count(); ++i) {
    cuteafd_dsv4_program_info_t info{};
    check(cuteafd_dsv4_program_info(i, &info) == 0);
    if (std::strcmp(info.name, "shared_relaxed_acceptance_stats_f32") == 0) {
      check(info.pointers == 6 && info.scalars == 3);
      check(std::strcmp(info.scalar_kinds, "iil") == 0);
      check(cuteafd_dsv4_program_load(i) == 0);
      return i;
    }
  }
  return UINT32_MAX;
}

struct StatsBuffers {
  cuteafd_device_buffer_t input{}, proposals{}, stats{}, ids{}, scores{}, status{};
  cuteafd_device_buffer_t old_ids{}, old_scores{};
  size_t rows, vocab, stride;

  StatsBuffers(size_t r, size_t v, size_t s) : rows(r), vocab(v), stride(s) {
    ok(cuteafd_alloc_device_buffer(rows * stride * 4, &input));
    for (auto* buffer : {&proposals, &ids, &scores, &status, &old_ids, &old_scores})
      ok(cuteafd_alloc_device_buffer(rows * 4, buffer));
    ok(cuteafd_alloc_device_buffer(rows * 3 * 4, &stats));
  }
  ~StatsBuffers() {
    for (auto* buffer : {&input, &proposals, &stats, &ids, &scores, &status, &old_ids, &old_scores})
      ok(cuteafd_free_device_buffer(buffer));
  }
  void fused(uint32_t program, void* stream) {
    void* pointers[] = {input.ptr, proposals.ptr, stats.ptr, ids.ptr, scores.ptr, status.ptr};
    const uint64_t scalars[] = {rows, vocab, stride};
    check(cuteafd_dsv4_program_launch(program, pointers, scalars, stream) == 0);
  }
  void legacy(void* stream) {
    // Legacy terminal has dense rows only; identity/timing cases use stride=vocab.
    check(stride == vocab);
    ok(cuteafd_cuda_logits_argmax_checked_f32_async(static_cast<const float*>(input.ptr),
       static_cast<uint32_t*>(old_ids.ptr), static_cast<float*>(old_scores.ptr), rows, vocab, stream));
  }
};

template <class T>
static std::vector<T> download(cuteafd_device_buffer_t buffer, size_t count) {
  std::vector<T> result(count);
  ok(cuteafd_copy_d2h(result.data(), buffer, count * sizeof(T)));
  return result;
}

static void stats_correctness(uint32_t program, void* stream) {
  for (size_t vocab : {size_t(1), size_t(257), size_t(4093), size_t(129280), size_t(248320)}) {
    constexpr size_t rows = 10;
    for (size_t padding : {size_t(0), size_t(3)}) {
      const size_t stride = vocab + padding;
      StatsBuffers b(rows, vocab, stride);
      std::vector<float> logits(rows * stride, std::numeric_limits<float>::quiet_NaN());
      std::vector<uint32_t> proposal(rows);
      for (size_t r = 0; r < rows; ++r) {
        proposal[r] = static_cast<uint32_t>((r * 17) % vocab);
        for (size_t c = 0; c < vocab; ++c) {
          float value = static_cast<float>(static_cast<int>((c * 31 + r * 7) % 1009) - 504) / 37.f;
          if (r == 0 || r == 5) value = 1.25f;
          if (r == 1) value = c % 2 == 0 ? 3e38f : -3e38f;
          if (r == 2) value = std::numeric_limits<float>::lowest();
          if (r == 3) value = c % 2 == 0 ? 1e-40f : -1e-40f;
          logits[r * stride + c] = value;
        }
      }
      proposal[4] = proposal[5] = proposal[8] = UINT32_MAX;
      proposal[9] = static_cast<uint32_t>(vocab);
      logits[6 * stride] = std::numeric_limits<float>::quiet_NaN();
      logits[7 * stride] = std::numeric_limits<float>::infinity();
      logits[8 * stride] = -std::numeric_limits<float>::infinity();
      ok(cuteafd_copy_h2d(b.input, logits.data(), logits.size() * 4));
      ok(cuteafd_copy_h2d(b.proposals, proposal.data(), rows * 4));
      b.fused(program, stream);
      ok(cuteafd_cuda_stream_synchronize(stream));
      const auto ids = download<uint32_t>(b.ids, rows);
      const auto scores = download<float>(b.scores, rows);
      const auto statuses = download<uint32_t>(b.status, rows);
      const auto stats = download<float>(b.stats, rows * 3);
      for (size_t r = 0; r < rows; ++r) {
        const uint32_t expected_status = r == 9 ? 3 : r >= 6 ? 2 : proposal[r] == UINT32_MAX ? 1 : 0;
        check(statuses[r] == expected_status);
        if (expected_status >= 2) {
          check(ids[r] == UINT32_MAX);
          check(expected_status == 2 ? std::isnan(scores[r]) : scores[r] == 0.f);
        } else {
          uint32_t winner = 0;
          for (size_t c = 1; c < vocab; ++c)
            if (logits[r * stride + c] > logits[r * stride + winner]) winner = c;
          check(ids[r] == winner && scores[r] == logits[r * stride + winner]);
          if (expected_status == 0) {
            const double maximum = logits[r * stride + winner];
            double mass = 0, moment = 0;
            for (size_t c = 0; c < vocab; ++c) {
              const double delta = double(logits[r * stride + c]) - maximum;
              const double weight = std::exp(delta);
              mass += weight;
              moment += weight * delta;
            }
            const double expected[] = {std::exp(double(logits[r * stride + proposal[r]]) - maximum) / mass,
                                       1.0 / mass, std::log(mass) - moment / mass};
            for (size_t j = 0; j < 3; ++j) {
              const double actual = stats[r * 3 + j];
              check(std::isfinite(actual));
              check(std::abs(actual - expected[j]) <= 3e-6 + 3e-5 * std::abs(expected[j]));
            }
          }
        }
        if (expected_status != 0)
          for (size_t j = 0; j < 3; ++j) check(stats[r * 3 + j] == 0.f);
      }
      const auto after = download<float>(b.input, logits.size());
      check(std::memcmp(after.data(), logits.data(), logits.size() * 4) == 0);
      if (padding == 0) {
        b.legacy(stream);
        ok(cuteafd_cuda_stream_synchronize(stream));
        const auto old_ids = download<uint32_t>(b.old_ids, rows);
        const auto old_scores = download<float>(b.old_scores, rows);
        for (size_t r = 0; r < 6; ++r) {
          check(ids[r] == old_ids[r]);
          check(std::memcmp(&scores[r], &old_scores[r], sizeof(float)) == 0);
        }
      }
    }
  }
  std::cout << "relaxed stats: float64 probability/entropy oracle, checked argmax identity, ties, "
               "extremes, status, mixed rows and padded strides passed\n";
}

template <class Launch>
static float elapsed_us(Launch launch, void* stream) {
  constexpr int iterations = 200;
  for (int i = 0; i < 10; ++i) launch();
  ok(cuteafd_cuda_stream_synchronize(stream));
  cudaEvent_t start{}, stop{};
  cuda_ok(cudaEventCreate(&start)); cuda_ok(cudaEventCreate(&stop));
  cuda_ok(cudaEventRecord(start, static_cast<cudaStream_t>(stream)));
  for (int i = 0; i < iterations; ++i) launch();
  cuda_ok(cudaEventRecord(stop, static_cast<cudaStream_t>(stream)));
  cuda_ok(cudaEventSynchronize(stop));
  float ms = 0;
  cuda_ok(cudaEventElapsedTime(&ms, start, stop));
  cuda_ok(cudaEventDestroy(stop)); cuda_ok(cudaEventDestroy(start));
  return ms * 1000.f / iterations;
}

static void stats_timing(uint32_t program, void* stream) {
  for (size_t rows : {size_t(1), size_t(16)}) {
    constexpr size_t vocab = 129280;
    StatsBuffers b(rows, vocab, vocab);
    std::vector<float> logits(rows * vocab);
    for (size_t i = 0; i < logits.size(); ++i)
      logits[i] = static_cast<float>(static_cast<int>(i % 1009) - 504) / 37.f;
    ok(cuteafd_copy_h2d(b.input, logits.data(), logits.size() * 4));
    for (int mode = 0; mode < 3; ++mode) {
      std::vector<uint32_t> proposals(rows, mode == 2 ? UINT32_MAX : 17);
      if (mode == 1) for (size_t r = 1; r < rows; r += 2) proposals[r] = UINT32_MAX;
      ok(cuteafd_copy_h2d(b.proposals, proposals.data(), rows * 4));
      b.fused(program, stream); b.legacy(stream);
      ok(cuteafd_cuda_stream_synchronize(stream));
      const auto ids = download<uint32_t>(b.ids, rows), old_ids = download<uint32_t>(b.old_ids, rows);
      const auto scores = download<float>(b.scores, rows), old_scores = download<float>(b.old_scores, rows);
      check(ids == old_ids);
      check(std::memcmp(scores.data(), old_scores.data(), rows * 4) == 0);
      const float old_us = elapsed_us([&] { b.legacy(stream); }, stream);
      const float fused_us = elapsed_us([&] { b.fused(program, stream); }, stream);
      std::cout << std::fixed << std::setprecision(3) << "relaxed_stats_timing rows=" << rows
                << " vocab=" << vocab << " mode=" << (mode == 0 ? "relaxed" : mode == 1 ? "mixed" : "argmax_only")
                << " legacy_us=" << old_us << " fused_us=" << fused_us
                << " delta_us=" << fused_us - old_us << "\n";
    }
  }
  std::cout << "all-exact runtime dispatch retains legacy terminal; no fused launch required\n";
}

static int relaxed_stats_test(bool timing) {
  const uint32_t program = stats_program();
  if (program == UINT32_MAX) {
    std::cerr << "shared_relaxed_acceptance_stats_f32 absent from this AOT build\n";
    return 77;
  }
  void* stream = nullptr; ok(cuteafd_cuda_stream_create(&stream));
  stats_correctness(program, stream);
  if (timing) stats_timing(program, stream);
  ok(cuteafd_cuda_stream_destroy(stream));
  return 0;
}
#endif

int main(int argc, char** argv) {
  if (argc == 2 && (std::strcmp(argv[1], "--relaxed-stats") == 0 ||
                    std::strcmp(argv[1], "--relaxed-stats-timing") == 0)) {
#ifdef CUTEAFD_TEST_GENERIC_AOT
    return relaxed_stats_test(std::strcmp(argv[1], "--relaxed-stats-timing") == 0);
#else
    std::cerr << "generic AOT programs are disabled in this build\n";
    return 77;
#endif
  }
  check(argc == 1);
  constexpr size_t rows = 7, vocab = 129280;
  std::vector<float> logits(rows*vocab, -10.f);
  logits[17] = logits[91] = -2.f;
  logits[vocab*1+vocab-1] = 5.f;
  for (size_t row = 2; row <= 4; ++row) logits[row*vocab+91] = 4.f;
  logits[vocab*2+113] = std::numeric_limits<float>::quiet_NaN();
  logits[vocab*3+113] = std::numeric_limits<float>::infinity();
  logits[vocab*4+113] = -std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < vocab; ++i) logits[vocab*5+i] = std::numeric_limits<float>::lowest();
  logits[vocab*6+255] = logits[vocab*6+256] = logits[vocab*6+65535] = 3.f;
  cuteafd_device_buffer_t input{}, indices{}, scores{};
  ok(cuteafd_alloc_device_buffer(logits.size()*4, &input));
  ok(cuteafd_alloc_device_buffer(rows*4, &indices));
  ok(cuteafd_alloc_device_buffer(rows*4, &scores));
  ok(cuteafd_copy_h2d(input, logits.data(), logits.size()*4));
  void* stream = nullptr; ok(cuteafd_cuda_stream_create(&stream));
  ok(cuteafd_cuda_logits_argmax_checked_f32_async(static_cast<const float*>(input.ptr),
     static_cast<uint32_t*>(indices.ptr), static_cast<float*>(scores.ptr), rows, vocab, stream));
  ok(cuteafd_cuda_stream_synchronize(stream));
  std::vector<uint32_t> actual(rows); std::vector<float> values(rows);
  ok(cuteafd_copy_d2h(actual.data(), indices, rows*4));
  ok(cuteafd_copy_d2h(values.data(), scores, rows*4));
  check(actual[0] == 17 && values[0] == -2.f);
  check(actual[1] == vocab-1 && values[1] == 5.f);
  for (size_t row = 2; row <= 4; ++row) check(std::isnan(values[row]));
  check(actual[5] == 0 && values[5] == std::numeric_limits<float>::lowest());
  check(actual[6] == 255 && values[6] == 3.f);
  ok(cuteafd_cuda_stream_destroy(stream));
  ok(cuteafd_free_device_buffer(&scores)); ok(cuteafd_free_device_buffer(&indices));
  ok(cuteafd_free_device_buffer(&input));
  std::cout << "checked argmax preserves ties and rejects every non-finite row\n";
}
