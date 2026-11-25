// tests/ik2_sampler_test.cpp - maya-ik2's sampler (src/ik2/ik2_sample.hpp) on the CPU:
//   1. dist() is softmax(logit / temp) over the top_p mass of the min(top_k, 1000) largest (ties: the lower token),
//      against a long double reference that sorts the whole row (to 1e-6 relative: the logit differences are floats);
//   2. from any candidate set that holds the 1000 largest and their ties (k::cand_topk's whole bins), dist() is the
//      whole row's, bit for bit; tokens past the 1000th are never in it, even at top_k 0 and top_p 1;
//   3. draw(), draw() without a token and pick() sample their distribution (chi-squared, fixed seeds);
//   4. verify_draft() - keep the drafter's argmax with p(draft), else draw p without it - gives p exactly, and keeps
//      the draft with probability p(draft);
//   5. the same seed gives the same tokens.
//     g++ -O2 -std=c++17 -Isrc/ik2 tests/ik2_sampler_test.cpp -o build/ik2_sampler_test
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <numeric>
#include <random>
#include <vector>

#include "ik2_sample.hpp"

using namespace ik2::sample;

static int g_fail = 0;
#define CHECK(c, ...)                                     \
    do {                                                  \
        if (!(c)) {                                       \
            ++g_fail;                                     \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                     \
            std::printf("\n");                            \
        }                                                 \
    } while (0)

constexpr int kK = 1000;   // the engine's kCandK

// the reference: the whole row sorted (logit desc, token asc), long double softmax, the same top_p rule
// (keep_at > 0: keep that many, the top_p rule left out; *mass_at: the mass of the keep_at largest, before the cut)
static std::vector<std::pair<int, long double>> ref_dist(const std::vector<float>& row, double temp, double top_p,
                                                         int top_k, int keep_at = 0, long double* mass_at = nullptr) {
    std::vector<int> ix(row.size());
    std::iota(ix.begin(), ix.end(), 0);
    std::stable_sort(ix.begin(), ix.end(), [&](int a, int b) { return row[a] > row[b]; });
    const int kk = std::min((int) row.size(), top_k > 0 ? std::min(top_k, kK) : kK);
    std::vector<long double> pr(kk);
    long double z = 0;
    for (int j = 0; j < kk; ++j) z += (pr[j] = std::exp(((long double) row[ix[j]] - row[ix[0]]) / temp));
    long double acc = 0;
    int keep = kk;
    for (int j = 0; j < kk; ++j) {
        acc += pr[j] / z;
        if (acc >= top_p) {
            keep = j + 1;
            break;
        }
    }
    if (keep_at > 0) {
        keep = std::min(keep_at, kk);
        if (mass_at) {
            *mass_at = 0;
            for (int j = 0; j < keep; ++j) *mass_at += pr[j] / z;
        }
    }
    long double zk = 0;
    for (int j = 0; j < keep; ++j) zk += pr[j];
    std::vector<std::pair<int, long double>> out(keep);
    for (int j = 0; j < keep; ++j) out[j] = {ix[j], pr[j] / zk};
    return out;
}

static std::vector<std::pair<float, int>> all_of(const std::vector<float>& row) {
    std::vector<std::pair<float, int>> cd(row.size());
    for (size_t v = 0; v < row.size(); ++v) cd[v] = {row[v], (int) v};
    return cd;
}

// chi-squared critical value at 1 - alpha (Wilson-Hilferty), alpha 1e-4: a fixed seed, so a pass is stable
static double chi2_crit(int df) {
    const double z = 3.719;   // the normal's 1 - 1e-4 quantile
    const double h = 2.0 / (9.0 * df);
    return df * std::pow(1.0 - h + z * std::sqrt(h), 3.0);
}
// chi-squared of counts against probabilities (cells with expected < 5 pooled)
static bool chi2_ok(const std::map<int, long>& got, const std::vector<std::pair<int, double>>& d, long n,
                    double* stat_out = nullptr, int* df_out = nullptr) {
    double stat = 0, pool_e = 0, pool_o = 0;
    int cells = 0;
    long seen = 0;
    for (const auto& e : d) {
        const double ex = e.second * n;
        auto it = got.find(e.first);
        const double ob = it == got.end() ? 0 : (double) it->second;
        seen += (long) ob;
        if (ex < 5) {
            pool_e += ex, pool_o += ob;
            continue;
        }
        stat += (ob - ex) * (ob - ex) / ex, ++cells;
    }
    if (pool_e > 0) stat += (pool_o - pool_e) * (pool_o - pool_e) / std::max(pool_e, 1e-9), ++cells;
    const int df = std::max(1, cells - 1);
    if (stat_out) *stat_out = stat;
    if (df_out) *df_out = df;
    return seen == n && stat < chi2_crit(df);   // (seen == n: nothing drawn outside d)
}

static std::vector<float> random_row(std::mt19937_64& rng, int V, int it) {
    std::normal_distribution<float> nd(0.0f, 3.0f);
    std::vector<float> row((size_t) V);
    for (float& x : row) x = nd(rng);
    for (int i = 0; i < 40; ++i) row[(size_t) (rng() % V)] = row[(size_t) (rng() % V)];   // (ties)
    if (it % 5 == 0) {   // (many ties at the top)
        const float top = *std::max_element(row.begin(), row.end());
        for (int i = 0; i < 30; ++i) row[(size_t) (rng() % V)] = top;
    }
    if (it % 7 == 0) {   // (ties across the 1000th: k::cand_topk keeps the whole bin)
        std::vector<float> s = row;
        std::nth_element(s.begin(), s.begin() + kK - 1, s.end(), std::greater<float>());
        const float kth = s[kK - 1];
        for (int i = 0; i < 25; ++i) row[(size_t) (rng() % V)] = kth;
    }
    if (it % 11 == 0) row[(size_t) (rng() % V)] = 0.0f, row[(size_t) (rng() % V)] = -0.0f;
    return row;
}

int main() {
    std::mt19937_64 rng(7);
    const double temps[] = {0.3, 0.6, 1.0, 1.3};
    const int ks[] = {0, 1, 5, 40, 64, 2000};
    const double ps[] = {1.0, 0.95, 0.8, 0.3};

    // 1 and 2: the distribution
    int cases = 0;
    for (int it = 0; it < 600; ++it) {
        const int V = it % 3 == 0 ? 1500 : 5000;
        const std::vector<float> row = random_row(rng, V, it);
        const double temp = temps[rng() % 4], top_p = ps[rng() % 4];
        const int top_k = ks[rng() % 6];

        auto cd = all_of(row);
        const auto d = dist(cd, temp, top_p, top_k, kK);
        auto r = ref_dist(row, temp, top_p, top_k);
        if (d.size() != r.size()) {
            // where the top_p cut falls is a rounding question only when the mass kept is top_p to 1e-12 (equal
            // logits at the cut), or when what is left out is below 1e-12 (top_p 1: the sum reaches 1 early)
            long double m = 0, m_ref = 0;
            const auto rk = ref_dist(row, temp, top_p, top_k, (int) d.size(), &m);
            ref_dist(row, temp, top_p, top_k, (int) r.size(), &m_ref);
            const bool tie_cut = std::fabs((double) m - top_p) < 1e-12 || std::fabs((double) m_ref - top_p) < 1e-12;
            const bool tail = std::fabs((double) (m_ref - m)) < 1e-12;
            CHECK(tie_cut || tail, "it %d: %zu kept against %zu (mass %.17Lg against %.17Lg, top_p %g)", it, d.size(),
                  r.size(), m, m_ref, top_p);
            r = rk;
        }
        double sum = 0;
        for (size_t j = 0; j < std::min(d.size(), r.size()); ++j) {
            CHECK(d[j].first == r[j].first, "it %d rank %zu: token %d against %d", it, j, d[j].first, r[j].first);
            CHECK(std::fabs(d[j].second - (double) r[j].second) <= 1e-12 + 1e-6 * (double) r[j].second,
                  "it %d rank %zu: p %.17g against %.17Lg", it, j, d[j].second, r[j].second);
            sum += d[j].second;
        }
        CHECK(std::fabs(sum - 1.0) < 1e-12, "it %d: sum %.17g", it, sum);
        CHECK((int) d.size() <= kK, "it %d: %zu kept, past %d", it, d.size(), kK);
        if (top_k == 0 && top_p >= 1.0 && V > kK) {   // the 1000 largest at most: the next one is never drawn
            std::vector<float> s2 = row;
            std::sort(s2.begin(), s2.end(), std::greater<float>());
            for (const auto& e : d) CHECK(row[e.first] >= s2[kK - 1], "it %d: token %d is past the %dth", it, e.first, kK);
        }

        // candidates: the 1000 largest, every tie of the 1000th, some smaller ones, shuffled
        std::vector<float> s = row;
        std::nth_element(s.begin(), s.begin() + kK - 1, s.end(), std::greater<float>());
        const float kth = s[kK - 1];
        std::vector<std::pair<float, int>> cand;
        for (int v = 0; v < V; ++v)
            if (row[v] >= kth || rng() % 17 == 0) cand.push_back({row[v], v});
        std::shuffle(cand.begin(), cand.end(), rng);
        const auto dc = dist(cand, temp, top_p, top_k, kK);
        bool same = dc.size() == d.size();
        for (size_t j = 0; same && j < d.size(); ++j) same = dc[j] == d[j];
        CHECK(same, "it %d: the candidates' distribution is not the whole row's", it);
        ++cases;
    }
    std::printf("dist: %d rows (whole row against the reference, candidates against the whole row)\n", cases);

    // 3: draw / draw without a token / pick, against their distribution
    for (int it = 0; it < 12; ++it) {
        const std::vector<float> row = random_row(rng, 3000, it);
        const double temp = temps[it % 4], top_p = it % 3 ? 0.95 : 1.0;
        const int top_k = it % 2 ? 40 : 64;
        auto cd = all_of(row);
        const auto d = dist(cd, temp, top_p, top_k, kK);
        const long n = 200000;
        std::mt19937_64 g(100 + it);
        std::map<int, long> c1, c2, c3;
        for (long i = 0; i < n; ++i) ++c1[draw(d, g)];
        double st;
        int df;
        bool ok = chi2_ok(c1, d, n, &st, &df);
        CHECK(ok, "draw it %d: chi2 %.1f df %d (crit %.1f)", it, st, df, chi2_crit(df));

        if (d.size() >= 2) {   // without the top token: the rest renormalised
            const int skip = d[0].first;
            std::vector<std::pair<int, double>> res;
            const double z = 1.0 - d[0].second;
            for (size_t j = 1; j < d.size(); ++j) res.push_back({d[j].first, d[j].second / z});
            for (long i = 0; i < n; ++i) ++c2[draw(d, g, skip)];
            ok = chi2_ok(c2, res, n, &st, &df);
            CHECK(ok, "draw-skip it %d: chi2 %.1f df %d", it, st, df);
            CHECK(c2.find(skip) == c2.end(), "draw-skip it %d: drew the skipped token", it);
        }
        for (long i = 0; i < n; ++i) {
            auto c = cd;
            ++c3[pick(c, temp, top_p, top_k, kK, g)];
        }
        ok = chi2_ok(c3, d, n, &st, &df);
        CHECK(ok, "pick it %d: chi2 %.1f df %d", it, st, df);
    }
    std::printf("draw, draw without a token, pick: 12 distributions x 200000 draws\n");

    // 4: speculative verification of an argmax draft
    for (int it = 0; it < 12; ++it) {
        const std::vector<float> row = random_row(rng, 3000, it + 50);
        auto cd = all_of(row);
        const auto d = dist(cd, temps[it % 4], 0.95, 64, kK);
        // the draft: the target's top token, a middle one, or one outside the kept set (p 0)
        const int draft = it % 3 == 0 ? d[0].first : it % 3 == 1 ? d[d.size() / 2].first : -7;
        double pd = 0;
        for (const auto& e : d)
            if (e.first == draft) pd = e.second;
        const long n = 300000;
        std::mt19937_64 g(900 + it);
        std::map<int, long> c;
        long kept_n = 0;
        for (long i = 0; i < n; ++i) {
            bool kept = false;
            const int t = verify_draft(d, draft, g, &kept);
            kept_n += kept;
            if (kept) CHECK(t == draft, "verify it %d: kept but returned %d", it, t);
            ++c[t];
        }
        double st;
        int df;
        const bool ok = chi2_ok(c, d, n, &st, &df);
        CHECK(ok, "verify it %d (draft p %.3f): chi2 %.1f df %d", it, pd, st, df);
        const double sd = std::sqrt(std::max(pd * (1 - pd), 1e-12) / n);
        CHECK(std::fabs((double) kept_n / n - pd) < 5 * sd + 1e-12, "verify it %d: kept %.5f against p %.5f", it,
              (double) kept_n / n, pd);
    }
    std::printf("verify_draft: 12 distributions x 300000 (argmax, middle and outside drafts)\n");

    // 5: the same seed, the same tokens
    {
        const std::vector<float> row = random_row(rng, 3000, 3);
        auto cd = all_of(row);
        const auto d = dist(cd, 1.0, 0.95, 64, kK);
        std::mt19937_64 a(42), b(42), c(43);
        int diff_other = 0;
        for (int i = 0; i < 1000; ++i) {
            bool k1, k2, k3;
            const int x = verify_draft(d, d[0].first, a, &k1), y = verify_draft(d, d[0].first, b, &k2);
            const int w = verify_draft(d, d[0].first, c, &k3);
            CHECK(x == y && k1 == k2, "seed: draw %d differs with the same seed", i);
            diff_other += x != w;
        }
        CHECK(diff_other > 0, "seed: another seed gave the same 1000 tokens");
    }

    std::printf(g_fail ? "ik2_sampler_test: %d FAILED\n" : "ik2_sampler_test: OK\n", g_fail);
    return g_fail ? 1 : 0;
}
