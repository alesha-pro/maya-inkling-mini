// tests/ik2_flash_bench.cu - the prompt's flash attention at a full (global) layer's shapes, timed: 24 heads, 8 K / V
// heads of 128, an FP8 or FP16 cache of nk keys, the last 512 queries of the prompt (a chunk of the engine's:
// Work::kRelbRows) against all of them, the relative bias made in the kernel (IK2_FA_BIAS) or from a table, the
// splits attn_flash_splits picks.  Prints ms and the tensor cores' TFLOPS (QK^T and PV, causal).
//   build/ik2_flash_bench [nk ...]        (default 8192 32768 65536 131072; ~0.6 GiB of VRAM at 131072)
#include "../src/ik2/ik2_kernels.cuh"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace k = ik2::k;

#define CU(x)                                                                             \
    do {                                                                                  \
        cudaError_t e_ = (x);                                                             \
        if (e_ != cudaSuccess) {                                                          \
            std::printf("CUDA %s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__); \
            std::exit(1);                                                                 \
        }                                                                                 \
    } while (0)

int main(int argc, char** argv) {
    std::vector<int> nks;
    for (int i = 1; i < argc; ++i) nks.push_back(std::atoi(argv[i]));
    if (nks.empty()) nks = {8192, 32768, 65536, 131072};
    const int H = 24, KV = 8, D = 128, Tq = 512, extent = 1024, KD = KV * D, HD = H * D;
    int n_sm = 0;
    CU(cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, 0));
    const int nk_max = *std::max_element(nks.begin(), nks.end());
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    // the cache: FP8 bytes (random, finite: exponent < 15) and FP16 values
    std::vector<uint8_t> k8((size_t) nk_max * KD), v8(k8.size());
    for (auto& b : k8) b = (uint8_t) ((rng() & 0x80) | (rng() % 0x70));
    for (auto& b : v8) b = (uint8_t) ((rng() & 0x80) | (rng() % 0x70));
    std::vector<uint16_t> k16(k8.size()), v16(k8.size());
    for (size_t i = 0; i < k16.size(); ++i)
        k16[i] = __half_as_ushort(__float2half_rn(nd(rng))), v16[i] = __half_as_ushort(__float2half_rn(nd(rng)));
    std::vector<uint16_t> q16((size_t) Tq * HD);
    for (auto& v : q16) v = __half_as_ushort(__float2half_rn(nd(rng) * 0.3f));
    std::vector<float> rq((size_t) Tq * H * 16), relp((size_t) extent * 16), sink((size_t) Tq * H);
    for (auto& v : rq) v = nd(rng);
    for (auto& v : relp) v = nd(rng) * 0.1f;
    for (auto& v : sink) v = nd(rng);
    uint8_t *dk8, *dv8;
    uint16_t *dk16, *dv16, *dq;
    float *drq, *drelp, *dsink, *dout, *dscr, *dbias;
    CU(cudaMalloc(&dk8, k8.size()));
    CU(cudaMalloc(&dv8, v8.size()));
    CU(cudaMalloc(&dk16, k16.size() * 2));
    CU(cudaMalloc(&dv16, v16.size() * 2));
    CU(cudaMemcpy(dk8, k8.data(), k8.size(), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dv8, v8.data(), v8.size(), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dk16, k16.data(), k16.size() * 2, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dv16, v16.data(), v16.size() * 2, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&dq, q16.size() * 2));
    CU(cudaMemcpy(dq, q16.data(), q16.size() * 2, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&drq, rq.size() * 4));
    CU(cudaMemcpy(drq, rq.data(), rq.size() * 4, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&drelp, relp.size() * 4));
    CU(cudaMemcpy(drelp, relp.data(), relp.size() * 4, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&dsink, sink.size() * 4));
    CU(cudaMemcpy(dsink, sink.data(), sink.size() * 4, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&dout, (size_t) Tq * HD * 4));
    CU(cudaMalloc(&dbias, (size_t) Tq * H * extent * 4));
    CU(cudaMemset(dbias, 0, (size_t) Tq * H * extent * 4));
    const size_t scr = (size_t) 16 * Tq * H * (D + 2);
    CU(cudaMalloc(&dscr, scr * 4));
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    const float scale = 1.0f / std::sqrt((float) D);
    for (int nk : nks) {
        const int qpos0 = nk - Tq;
        const int ns = k::attn_flash_splits(Tq, H, nk, n_sm);
        // causal flops: each query its keys, QK^T and PV, 2 flops a multiply-add
        const double fl = 4.0 * H * D * ((double) Tq * qpos0 + (double) Tq * (Tq + 1) / 2);
        for (int v = 0; v < 4; ++v) {
            const int fp8 = v < 2, fast = v & 1;
            k::attn_flash_fast(fast);
            auto run = [&] {
                if (fp8)
                    k::attn_flash_fp8(dq, HD, dk8, dv8, 0, H, KV, D, Tq, qpos0, scale * 0.0625f, 1.0f / 32, nullptr,
                                      extent, 1 << 30, dsink, dout, HD, dscr, scr, ns, 0, nullptr, drq, drelp);
                else
                    k::attn_flash_f16(dq, HD, dk16, dv16, 0, H, KV, D, Tq, qpos0, scale, nullptr, extent, 1 << 30,
                                      dsink, dout, HD, dscr, scr, ns, 0, nullptr, drq, drelp);
            };
            run();
            CU(cudaDeviceSynchronize());
            std::vector<float> ts;
            for (int r = 0; r < 7; ++r) {
                cudaEventRecord(e0);
                run();
                cudaEventRecord(e1);
                CU(cudaEventSynchronize(e1));
                float ms;
                cudaEventElapsedTime(&ms, e0, e1);
                ts.push_back(ms);
            }
            std::sort(ts.begin(), ts.end());
            const float ms = ts[ts.size() / 2];
            std::printf("nk %6d  %s %s  splits %2d  %8.3f ms  %5.1f TFLOPS\n", nk, fp8 ? "FP8 " : "FP16",
                        fast ? "fast" : "    ", ns, ms, fl / ms * 1e-9);
        }
    }
    return 0;
}
