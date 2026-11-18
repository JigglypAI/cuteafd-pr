// Sparse MLA prefill kernels (glm_mla_prefill.cu) against an F32 host reference (no TF32) on
// synthetic FP8 latent records, and their times.
//   glm_mla_prefill_selftest [rows] [record_bytes 656|528] [heads] [iterations] [kernels, e.g. 12] [mixed|spread|medium|peaked] [topk]
// Records: latent rows N(0, 1) times per-channel and per-record magnitudes, E4M3 codes with
// one FP32 scale (amax / 448) per 128 channels; queries mix noise with the latents of a few
// of the row's selected records (peaked heads) at per-head magnitudes. Each kernel's error
// (max abs, relative L2) over sampled rows against the reference, then the median time of
// [iterations] launches over every row.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

extern "C" int32_t cuteafd_glm_mla_prefill_lse(const void* q, const void* kv, const void* indices, const void* lengths,
                                           void* out, int32_t rows, int32_t heads, int32_t topk,
                                           int32_t record_bytes, float scale_log2, int32_t kernel, void* stream, float* lse);

extern "C" int32_t cuteafd_glm_mla_prefill_compare(const void* q, const void* kv, const void* indices,
                                                 const void* lengths, int32_t rows, int32_t heads, int32_t topk,
                                                 int32_t record_bytes, float scale_log2, double* stats, void* stream);

#define CHECK(x)                                                                      \
  do {                                                                                \
    cudaError_t e_ = (x);                                                             \
    if (e_ != cudaSuccess) {                                                          \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      std::exit(1);                                                                   \
    }                                                                                 \
  } while (0)

static float bf16_to_float(uint16_t v) {
  uint32_t bits = uint32_t(v) << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}
static uint16_t float_to_bf16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  return uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
static float e4m3_to_float(uint8_t v) {
  __nv_fp8_e4m3 x;
  x.__x = v;
  return float(x);
}

int main(int argc, char** argv) {
  const int rows = argc > 1 ? std::atoi(argv[1]) : 4096;
  const int rec = argc > 2 ? std::atoi(argv[2]) : 656;
  const int heads = argc > 3 ? std::atoi(argv[3]) : 64;
  const int iterations = argc > 4 ? std::atoi(argv[4]) : 10;
  const char* kernels = argc > 5 ? argv[5] : (rec == 656 ? "012" : "01");
  const char* pattern = argc > 6 ? argv[6] : "mixed";
  const int topk = argc > 7 ? std::atoi(argv[7]) : (rec == 656 ? 2048 : 2112);
  const int slots = 12288, dq = rec == 656 ? 576 : 512;
  if (rows < 1 || iterations < 1 || topk < 1 || topk >= slots || heads < 32 || heads % 32 ||
      (rec != 656 && rec != 528) || (std::strcmp(pattern, "mixed") && std::strcmp(pattern, "spread") &&
      std::strcmp(pattern, "medium") && std::strcmp(pattern, "peaked"))) return 2;
  const bool rope = rec == 656;
  const float scale = 1.0f / std::sqrt(rope ? 256.0f : 256.0f);
  const float scale_log2 = scale * 1.4426950408889634f;
  std::mt19937_64 rng(1234);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

  // Records and their exact dequantized latents (codes x scale) / RoPE dims.
  std::vector<float> channel(512);
  for (auto& c : channel) c = std::exp(0.5f * normal(rng)) * (uniform(rng) < 0.02f ? 6.0f : 1.0f);
  std::vector<uint8_t> kv(size_t(slots) * rec);
  std::vector<float> k(size_t(slots) * dq);
  for (int t = 0; t < slots; ++t) {
    const float magnitude = std::exp(0.7f * normal(rng));
    float x[512];
    for (int c = 0; c < 512; ++c) x[c] = normal(rng) * channel[c] * magnitude;
    uint8_t* r = &kv[size_t(t) * rec];
    for (int gi = 0; gi < 4; ++gi) {
      float amax = 0;
      for (int c = 0; c < 128; ++c) amax = std::max(amax, std::fabs(x[gi * 128 + c]));
      const float s = amax / 448.0f;
      std::memcpy(r + 512 + 4 * gi, &s, 4);
      for (int c = 0; c < 128; ++c) {
        const __nv_fp8_e4m3 code(x[gi * 128 + c] / s);
        r[gi * 128 + c] = code.__x;
        k[size_t(t) * dq + gi * 128 + c] = e4m3_to_float(code.__x) * s;
      }
    }
    if (rope) {
      for (int d = 0; d < 64; ++d) {
        const uint16_t b = float_to_bf16(normal(rng) * magnitude);
        std::memcpy(r + 528 + 2 * d, &b, 2);
        k[size_t(t) * dq + 512 + d] = bf16_to_float(b);
      }
    }
  }
  // Selections: most rows full, some short; ~1% masked entries.
  std::vector<int32_t> indices(size_t(rows) * topk, -1), lengths(rows);
  // Slot zero is never selected, so poisoning it tests masked-record isolation.
  std::vector<int32_t> perm(slots - 1);
  for (int i = 0; i < slots - 1; ++i) perm[i] = i + 1;
  for (int row = 0; row < rows; ++row) {
    lengths[row] = uniform(rng) < 0.8f ? topk : 1 + int(uniform(rng) * (topk - 1));
    for (int i = 0; i < lengths[row]; ++i) std::swap(perm[i], perm[i + int(uniform(rng) * (slots - 1 - i))]);
    for (int i = 0; i < lengths[row]; ++i) indices[size_t(row) * topk + i] = uniform(rng) < 0.01f ? -1 : perm[i];
  }
  if (rows > 1) lengths[0] = 0;
  if (rows > 2) std::fill_n(indices.begin() + topk, topk, -1);
  // Queries: noise plus a few selected records' latents (peaked heads), per-head magnitudes.
  std::vector<uint16_t> q(size_t(rows) * heads * dq);
  for (int row = 0; row < rows; ++row) {
    for (int h = 0; h < heads; ++h) {
      const float sigma = 0.3f * std::exp(0.8f * normal(rng));
      const int peaks = h % 4;
      std::vector<float> v(dq);
      for (int d = 0; d < dq; ++d) v[d] = sigma * normal(rng);
      for (int p = 0; p < peaks && lengths[row] > 0; ++p) {
        const int slot = indices[size_t(row) * topk + int(uniform(rng) * lengths[row])];
        if (slot < 0) continue;
        double norm = 0;
        for (int d = 0; d < dq; ++d) norm += double(k[size_t(slot) * dq + d]) * k[size_t(slot) * dq + d];
        const float w = (4.0f + 8.0f * uniform(rng)) / std::sqrt(float(norm) + 1e-12f) / scale;
        for (int d = 0; d < dq; ++d) v[d] += w * k[size_t(slot) * dq + d] / std::sqrt(float(dq));
      }
      if (std::strcmp(pattern, "mixed")) {
        // Normalize against the exact record pool's per-coordinate RMS, keeping score entropy controlled.
        for (int d = 0; d < dq; ++d) {
          const float rms = d < 512 ? channel[d] * 1.65f : 1.65f;
          const float sigma = !std::strcmp(pattern, "spread") ? 0.03f : 2.0f;
          v[d] = normal(rng) * sigma / rms;
        }
        if (!std::strcmp(pattern, "peaked") && lengths[row] > 0) {
          int slot = -1;
          for (int i = 0; i < lengths[row] && slot < 0; ++i) slot = indices[size_t(row) * topk + i];
          if (slot >= 0) {
            float norm = 0;
            for (int d = 0; d < dq; ++d) norm += k[size_t(slot) * dq + d] * k[size_t(slot) * dq + d];
            for (int d = 0; d < dq; ++d) v[d] = 24.0f * k[size_t(slot) * dq + d] / (scale * norm);
          }
        }
      }
      for (int d = 0; d < dq; ++d) q[(size_t(row) * heads + h) * dq + d] = float_to_bf16(v[d]);
    }
  }

  void *dq_, *dkv, *dind, *dlen, *dout;
  float* dlse;
  const size_t out_elems = size_t(rows) * heads * 512;
  CHECK(cudaMalloc(&dq_, q.size() * 2));
  CHECK(cudaMalloc(&dkv, kv.size()));
  CHECK(cudaMalloc(&dind, indices.size() * 4));
  CHECK(cudaMalloc(&dlen, lengths.size() * 4));
  CHECK(cudaMalloc(&dout, out_elems * 2));
  CHECK(cudaMalloc(&dlse, size_t(rows) * heads * sizeof(float)));
  CHECK(cudaMemcpy(dq_, q.data(), q.size() * 2, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dkv, kv.data(), kv.size(), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dind, indices.data(), indices.size() * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dlen, lengths.data(), lengths.size() * 4, cudaMemcpyHostToDevice));

  // Scalar F32 reference on exact decoded records, never a BF16-rounded K/V or TF32 matmul.
  std::vector<int> sample;
  for (int i = 0; i < std::min(rows, 24); ++i) sample.push_back(i < 2 ? i : int((size_t(i) * 7919 + 13) % rows));
  std::vector<float> ref(sample.size() * heads * 512), ref_lse(sample.size() * heads);
  {
    std::vector<std::thread> pool;
    const int workers = std::max(1u, std::thread::hardware_concurrency());
    for (int w = 0; w < workers; ++w) {
      pool.emplace_back([&, w] {
        for (size_t job = w; job < sample.size() * heads; job += workers) {
          const int si = int(job / heads), h = int(job % heads), row = sample[si];
          std::vector<float> sc(lengths[row]);
          float mx = -INFINITY;
          for (int i = 0; i < lengths[row]; ++i) {
            const int slot = indices[size_t(row) * topk + i];
            if (slot < 0) {
              sc[i] = -INFINITY;
              continue;
            }
            float dot = 0;
            for (int d = 0; d < dq; ++d)
              dot += bf16_to_float(q[(size_t(row) * heads + h) * dq + d]) * k[size_t(slot) * dq + d];
            sc[i] = dot * scale;
            mx = std::max(mx, sc[i]);
          }
          float l = 0;
          std::vector<float> o(512, 0.0f);
          for (int i = 0; i < lengths[row]; ++i) {
            const int slot = indices[size_t(row) * topk + i];
            if (slot < 0) continue;
            const float p = std::exp(sc[i] - mx);
            l += p;
            for (int c = 0; c < 512; ++c) o[c] += p * k[size_t(slot) * dq + c];
          }
          ref_lse[size_t(si) * heads + h] = l > 0 ? mx + std::log(l) : -INFINITY;
          for (int c = 0; c < 512; ++c) ref[(size_t(si) * heads + h) * 512 + c] = l > 0 ? o[c] / l : 0;
        }
      });
    }
    for (auto& t : pool) t.join();
  }

  std::vector<uint16_t> out(out_elems), base;
  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));
  cudaEvent_t a, b;
  CHECK(cudaEventCreate(&a));
  CHECK(cudaEventCreate(&b));
  const char* names[] = {"f16", "e4m3-q2p2", "bf16"};
  std::vector<float> lse(size_t(rows) * heads);
  int failures = 0;
  for (int kernel = 0; kernel < 3; ++kernel) {
    if (!std::strchr(kernels, '0' + kernel)) continue;
    CHECK(cudaMemset(dout, 0xFF, out_elems * 2));
    CHECK(cudaError_t(cuteafd_glm_mla_prefill_lse(dq_, dkv, dind, dlen, dout, rows, heads, topk, rec, scale_log2,
                                              kernel, stream, dlse)));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaMemcpy(out.data(), dout, out_elems * 2, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(lse.data(), dlse, lse.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double max_abs = 0, err2 = 0, ref2 = 0, base_err2 = 0, base_ref2 = 0, lse_max = 0;
    bool finite = true;
    for (size_t si = 0; si < sample.size(); ++si) {
      for (int h = 0; h < heads; ++h) {
        const float expected_lse = ref_lse[si * heads + h], actual_lse = lse[size_t(sample[si]) * heads + h];
        if (expected_lse == -INFINITY) finite &= actual_lse == -INFINITY;
        else {
          finite &= std::isfinite(actual_lse);
          lse_max = std::max(lse_max, double(std::fabs(actual_lse - expected_lse)));
        }
        for (int c = 0; c < 512; ++c) {
          const double x = bf16_to_float(out[(size_t(sample[si]) * heads + h) * 512 + c]);
          const double y = ref[(si * heads + h) * 512 + c];
          finite &= std::isfinite(x);
          max_abs = std::max(max_abs, std::fabs(x - y));
          err2 += (x - y) * (x - y);
          ref2 += y * y;
        }
      }
    }
    if (kernel == 0) {
      base = out;
    } else if (!base.empty()) {
      for (size_t i = 0; i < out_elems; ++i) {
        const double x = bf16_to_float(out[i]), y = bf16_to_float(base[i]);
        base_err2 += (x - y) * (x - y);
        base_ref2 += y * y;
      }
    }
    // Determinism: a second launch must match bit for bit.
    std::vector<uint16_t> again(out_elems);
    CHECK(cudaError_t(cuteafd_glm_mla_prefill_lse(dq_, dkv, dind, dlen, dout, rows, heads, topk, rec, scale_log2,
                                              kernel, stream, dlse)));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaMemcpy(again.data(), dout, out_elems * 2, cudaMemcpyDeviceToHost));
    const bool same = std::memcmp(again.data(), out.data(), out_elems * 2) == 0;
    // Rebase the final row onto a one-row launch; it must keep the same arithmetic.
    const int last = rows - 1;
    CHECK(cudaError_t(cuteafd_glm_mla_prefill_lse(static_cast<uint16_t*>(dq_) + size_t(last) * heads * dq,
        dkv, static_cast<int32_t*>(dind) + size_t(last) * topk, static_cast<int32_t*>(dlen) + last,
        dout, 1, heads, topk, rec, scale_log2, kernel, stream, dlse)));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaMemcpy(again.data(), dout, size_t(heads) * 512 * 2, cudaMemcpyDeviceToHost));
    const bool row_invariant = std::memcmp(again.data(), out.data() + size_t(last) * heads * 512,
                                          size_t(heads) * 512 * 2) == 0;
    CHECK(cudaMemsetAsync(dkv, 0x7f, rec, stream));
    CHECK(cudaError_t(cuteafd_glm_mla_prefill_lse(dq_, dkv, dind, dlen, dout, rows, heads, topk, rec, scale_log2,
                                                kernel, stream, dlse)));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaMemcpy(again.data(), dout, out_elems * 2, cudaMemcpyDeviceToHost));
    const bool poison_safe = std::memcmp(again.data(), out.data(), out_elems * 2) == 0;
    CHECK(cudaMemcpyAsync(dkv, kv.data(), rec, cudaMemcpyHostToDevice, stream));
    CHECK(cudaStreamSynchronize(stream));
    std::vector<float> times;
    for (int it = 0; it < iterations + 2; ++it) {
      CHECK(cudaEventRecord(a, stream));
      CHECK(cudaError_t(cuteafd_glm_mla_prefill_lse(dq_, dkv, dind, dlen, dout, rows, heads, topk, rec, scale_log2, kernel, stream, nullptr)));
      CHECK(cudaEventRecord(b, stream));
      CHECK(cudaEventSynchronize(b));
      float ms;
      CHECK(cudaEventElapsedTime(&ms, a, b));
      if (it >= 2) times.push_back(ms);
    }
    std::sort(times.begin(), times.end());
    const double rel = ref2 > 0 ? std::sqrt(err2 / ref2) : std::sqrt(err2);
    std::printf("%-10s rec %d heads %d rows %d: max abs err %.3e rel RMS %.3e vs F32%s LSE max %.3e | rel L2 vs f16 %.3e | "
                "deterministic %s | %.3f ms (min %.3f)\n",
                names[kernel], rec, heads, rows, max_abs, rel, finite ? "" : " NON-FINITE", lse_max,
                base_ref2 > 0 ? std::sqrt(base_err2 / base_ref2) : 0.0, same ? "yes" : "NO", times[times.size() / 2],
                times[0]);
    std::printf("  row invariant %s | slot-zero poison %s\n", row_invariant ? "yes" : "NO",
                poison_safe ? "yes" : "NO");
    failures += !finite || !same || !row_invariant || !poison_safe || rel > 0.05 || (kernel == 2 && lse_max > 3e-4);
  }
  double stats[9];
  std::fill_n(stats, 9, -1.0);
  CHECK(cudaError_t(cuteafd_glm_mla_prefill_compare(dq_, dkv, dind, dlen, rows, heads, topk,
                                                rec, scale_log2, stats, stream)));
  const bool diagnostic = std::isfinite(stats[0]) && stats[0] >= 0 && stats[3] == 0
      && std::isfinite(stats[5]) && stats[5] >= 0
      && (rope ? std::isfinite(stats[6]) && stats[6] >= 0 && std::abs(stats[8] - stats[5]) <= 1e-12 * stats[5]
               : stats[6] == -1.0 && stats[7] == -1.0 && stats[8] == -1.0);
  std::printf("diagnostic includes exactly the supported kernels: %s\n", diagnostic ? "yes" : "NO");
  failures += !diagnostic;
  CHECK(cudaEventDestroy(a));
  CHECK(cudaEventDestroy(b));
  CHECK(cudaStreamDestroy(stream));
  CHECK(cudaFree(dq_)); CHECK(cudaFree(dkv)); CHECK(cudaFree(dind)); CHECK(cudaFree(dlen));
  CHECK(cudaFree(dout)); CHECK(cudaFree(dlse));
  return failures ? 1 : 0;
}
