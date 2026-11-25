// src/ik2/ik2_sample.hpp - maya-ik2's sampling on the host, over a row's candidates (logit, token): no CUDA, so
// tests/ik2_sampler_test.cpp checks it on the CPU.
//
// The distribution: softmax(logit / temp) over the top_p mass of the min(top_k, kmax) largest candidates (top_k 0:
// kmax; ties: the lower token first), renormalised.  It is the distribution over the candidates given, so it is the
// whole row's only when they hold its kmax largest logits - the engine's k::cand_topk keeps every logit of the
// highest bins that hold kmax (kCandK = 1000), and the CLI and the serve loop fall back to the whole row otherwise.
// Tokens past the kmax largest are never drawn, even at top_p 1: with top_k 0 the engine samples from the top 1000.
// The server sends top_k 64 when the request leaves it at 0 (serve/server.py), so through the API it is the top 64.
#pragma once

#include <algorithm>
#include <cmath>
#include <random>
#include <utility>
#include <vector>

namespace ik2::sample {

// cd: (logit, token); sorted in place - its kept part first, largest first.  Returns (token, probability), summing to 1.
inline std::vector<std::pair<int, double>> dist(std::vector<std::pair<float, int>>& cd, double temp, double top_p,
                                                int top_k, int kmax) {
    const int kk = std::min((int) cd.size(), top_k > 0 ? std::min(top_k, kmax) : kmax);
    std::partial_sort(cd.begin(), cd.begin() + kk, cd.end(),
                      [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                          return a.first > b.first || (a.first == b.first && a.second < b.second);
                      });
    std::vector<double> pr(kk);
    double z = 0;
    for (int j = 0; j < kk; ++j) z += (pr[j] = std::exp((cd[j].first - cd[0].first) / temp));
    double acc = 0;
    int keep = kk;
    for (int j = 0; j < kk; ++j) {
        acc += pr[j] / z;
        if (acc >= top_p) {
            keep = j + 1;
            break;
        }
    }
    double zk = 0;
    for (int j = 0; j < keep; ++j) zk += pr[j];
    std::vector<std::pair<int, double>> out(keep);
    for (int j = 0; j < keep; ++j) out[j] = {cd[j].second, pr[j] / zk};
    return out;
}

// One draw from d, leaving out the token skip (-1: none): the residual distribution, renormalised.
inline int draw(const std::vector<std::pair<int, double>>& d, std::mt19937_64& rng, int skip = -1) {
    double z = 0;
    for (const auto& e : d)
        if (e.first != skip) z += e.second;
    if (z <= 0) return d[0].first;
    double u = std::uniform_real_distribution<double>(0, z)(rng);
    for (const auto& e : d) {
        if (e.first == skip) continue;
        u -= e.second;
        if (u <= 0) return e.first;
    }
    for (auto it = d.rbegin(); it != d.rend(); ++it)
        if (it->first != skip) return it->first;
    return d[0].first;
}

// One token straight from the candidates, as dist then draw would (one uniform; the plain decode paths).
inline int pick(std::vector<std::pair<float, int>>& cd, double temp, double top_p, int top_k, int kmax,
                std::mt19937_64& rng) {
    const int kk = std::min((int) cd.size(), top_k > 0 ? std::min(top_k, kmax) : kmax);
    std::partial_sort(cd.begin(), cd.begin() + kk, cd.end(),
                      [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                          return a.first > b.first || (a.first == b.first && a.second < b.second);
                      });
    std::vector<double> pr(kk);
    double z = 0;
    for (int j = 0; j < kk; ++j) z += (pr[j] = std::exp((cd[j].first - cd[0].first) / temp));
    double acc = 0;
    int keep = kk;
    for (int j = 0; j < kk; ++j) {
        acc += pr[j] / z;
        if (acc >= top_p) {
            keep = j + 1;
            break;
        }
    }
    double zk = 0;
    for (int j = 0; j < keep; ++j) zk += pr[j];
    double u = std::uniform_real_distribution<double>(0, zk)(rng);
    for (int j = 0; j < keep; ++j) {
        u -= pr[j];
        if (u <= 0) return cd[j].second;
    }
    return cd[keep - 1].second;
}

// A speculative step's verdict on one draft: the drafter's argmax (a point mass), kept with the target's
// probability of it, else a draw from the target without it.  The result is distributed as d exactly.  Returns the
// token; *kept says whether it is the draft.
inline int verify_draft(const std::vector<std::pair<int, double>>& d, int draft, std::mt19937_64& rng, bool* kept) {
    double pd = 0;
    for (const auto& e : d)
        if (e.first == draft) pd = e.second;
    if (std::uniform_real_distribution<double>(0, 1)(rng) < pd) {
        *kept = true;
        return draft;
    }
    *kept = false;
    return draw(d, rng, draft);
}

}   // namespace ik2::sample
