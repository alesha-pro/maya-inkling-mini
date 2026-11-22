// tests/ik2_expert_bench.cu - P3-4's speed gate: the grouped NVFP4 expert GEMM (maya-ik2's grouped path) timed at
// the model's shapes - C = 2304 input / F = 1152 expert FFN width, 8 tokens a group - and its output dumped, so the
// emulated path (IK2_GROUP_MMA=3 IK2_EXPERT_WMMA=0) and the gfx12 WMMA path (IK2_GROUP_MMA=3 IK2_EXPERT_WMMA=1)
// can be compared: the same shapes, the same weights, the same activations, both from one binary.
//
//   ./ik2_expert_bench gate 40 /tmp/gate.bin
//   ./ik2_expert_bench down 40 /tmp/down.bin
#include "../src/ik2/ik2_kernels.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = ik2::k;

namespace {
int g_fail = 0;
std::mt19937 rng(20261121);

#define CU(x)                                                                                  \
    do {                                                                                       \
        cudaError_t e_ = (x);                                                                  \
        if (e_ != cudaSuccess) {                                                               \
            std::printf("CUDA %s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__);      \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (0)

template <class T>
T* dev(const std::vector<T>& h) {
    T* p = nullptr;
    CU(cudaMalloc(&p, std::max<size_t>(1, h.size()) * sizeof(T)));
    if (!h.empty()) CU(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

// an NVFP4 expert matrix: 36-byte blocks of 64 values (4 ue4m3 sub-block scales, then 32 code bytes)
std::vector<uint8_t> rand_nvfp4(size_t n_values) {
    const size_t blocks = n_values / 64;
    std::vector<uint8_t> v(blocks * 36);
    for (size_t b = 0; b < blocks; ++b) {
        for (int i = 0; i < 4; ++i) v[b * 36 + i] = (uint8_t) (0x38 | (rng() & 7));   // (e = 7: 2^-1..2^0)
        for (int i = 4; i < 36; ++i) v[b * 36 + i] = (uint8_t) rng();
    }
    return v;
}
}   // namespace

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "gate";
    const bool gate = std::strcmp(mode, "gate") == 0 || std::strcmp(mode, "gate1") == 0;
    const bool single = std::strcmp(mode, "gate1") == 0 || std::strcmp(mode, "down1") == 0;   // (the one-token path)
    const int iters = argc > 2 ? std::atoi(argv[2]) : 20;
    const char* out_path = argc > 3 ? argv[3] : nullptr;
    const int C = 2304, F = 1152, G = 128, K = k::kGroupMax;   // (the model's shapes: 128 experts, 8 tokens a group)
    const int S = G * K;
    const size_t stride_g = (size_t) (C / 64) * 36, stride_d = (size_t) (F / 64) * 36;
    const size_t eb0 = (size_t) F * stride_g, eb1 = eb0, eb2 = (size_t) C * stride_d;
    const size_t slot = (eb0 + eb1 + eb2 + 255) / 256 * 256, E = 8;   // (8 distinct expert blobs, read round-robin)
    std::vector<uint8_t> pool(slot * E);
    for (size_t e = 0; e < E; ++e) {
        auto g = rand_nvfp4((size_t) F * C), u = rand_nvfp4((size_t) F * C), d = rand_nvfp4((size_t) C * F);
        std::memcpy(pool.data() + e * slot, g.data(), g.size());
        std::memcpy(pool.data() + e * slot + eb0, u.data(), u.size());
        std::memcpy(pool.data() + e * slot + eb0 + eb1, d.data(), d.size());
    }
    uint8_t* dpool = dev(pool);
    std::vector<const uint8_t*> blob(G);
    for (int g = 0; g < G; ++g) blob[g] = dpool + (g % E) * slot;
    std::vector<int> n(G), tok(S), row(S);
    for (int g = 0; g < G; ++g) {
        n[g] = K;
        for (int i = 0; i < K; ++i) tok[(size_t) g * K + i] = g * K + i, row[(size_t) g * K + i] = g * K + i;
    }
    std::vector<float> sg(S, 1.0f), su(S, 1.0f);
    std::vector<uint16_t> input(S * (size_t) C), h(S * (size_t) F), input2(2 * (size_t) C);   // (input2: the gate limb pair)
    for (auto& x : input) {
        const float v = (float) (rng() % 2001) / 1000.0f - 1.0f;
        x = (uint16_t) __half_as_ushort(__float2half_rn(v));
    }
    for (int i = 0; i < C; ++i) input2[i] = input[i], input2[C + i] = input[i];   // (hi limb, lo limb)
    std::vector<float> y((size_t) S * C, 0.0f);

    uint8_t* dblob_pool = dpool;
    const uint8_t** dblob = nullptr;
    const int slots = std::max(G, S);   // (the one-token path has a slot a token, the grouped path a group)
    CU(cudaMalloc(&dblob, slots * sizeof(uint8_t*)));
    std::vector<const uint8_t*> hblob(slots);
    for (int i = 0; i < slots; ++i) hblob[i] = blob[i % G];
    CU(cudaMemcpy(dblob, hblob.data(), slots * sizeof(uint8_t*), cudaMemcpyHostToDevice));
    (void) dblob_pool;
    int* dn = dev(n);
    int* dtok = dev(tok);
    int* drow = dev(row);
    float* dsg = dev(sg);
    float* dsu = dev(su);
    uint16_t* dinput = dev(input);
    uint16_t* dinput2 = dev(input2);
    uint16_t* dh = dev(h);
    float* dy = dev(y);

    cudaStream_t s = nullptr;
    auto once = [&] {
        if (single) {
            if (gate)
                ik2::k::experts_gate_up_swiglu_mma(dblob, S, eb0, dinput2, C, F, dsg, dsu, 20.0f, 20.0f, dh, s);
            else
                ik2::k::experts_down_mma(dblob, S, eb0 + eb1, dh, C, F, dy, s);
        } else if (gate) {
            ik2::k::experts_gate_up_swiglu_g(dblob, dn, dtok, drow, G, k::kNVFP4, k::kNVFP4, eb0, nullptr, C, F, dsg,
                                             dsu, 20.0f, 20.0f, dh, s, dinput);
        } else {
            ik2::k::experts_down_g(dblob, dn, drow, G, k::kNVFP4, eb0 + eb1, dh, C, F, dy, s);
        }
    };
    once();   // (warm up: JIT, the first page-ins)
    CU(cudaDeviceSynchronize());
    std::vector<float> ms;
    for (int i = 0; i < iters; ++i) {
        cudaEvent_t a, b;
        CU(cudaEventCreate(&a));
        CU(cudaEventCreate(&b));
        CU(cudaEventRecord(a, s));
        once();
        CU(cudaEventRecord(b, s));
        CU(cudaEventSynchronize(b));
        float t = 0;
        CU(cudaEventElapsedTime(&t, a, b));
        ms.push_back(t);
        cudaEventDestroy(a);
        cudaEventDestroy(b);
    }
    std::sort(ms.begin(), ms.end());
    const double best = ms.front(), med = ms[ms.size() / 2];
    const double dots = single ? (double) S : (double) G * K;   // (dots: gate + up counted as two)
    const double flops = gate ? 2.0 * dots * (double) F * 2 * C : 2.0 * slots * (double) C * F;
    std::printf("ik2_expert_bench %s C=%d F=%d %s: best %.1f us, median %.1f us, %.0f GFLOP/s\n", mode, C, F,
                single ? "one token a slot" : "8 tokens a group", best * 1e3, med * 1e3,
                flops / (med * 1e-3) / 1e9);
    if (out_path) {
        std::vector<uint8_t> raw;
        if (gate) {
            std::vector<uint16_t> out(h.size());
            CU(cudaMemcpy(out.data(), dh, out.size() * 2, cudaMemcpyDeviceToHost));
            raw.resize(out.size() * 2);
            std::memcpy(raw.data(), out.data(), raw.size());
        } else {
            std::vector<float> out(y.size());
            CU(cudaMemcpy(out.data(), dy, out.size() * 4, cudaMemcpyDeviceToHost));
            raw.resize(out.size() * 4);
            std::memcpy(raw.data(), out.data(), raw.size());
        }
        FILE* f = std::fopen(out_path, "wb");
        if (!f) return 2;
        std::fwrite(raw.data(), 1, raw.size(), f);
        std::fclose(f);
        std::printf("wrote %s (%zu bytes)\n", out_path, raw.size());
    }
    return g_fail ? 1 : 0;
}
