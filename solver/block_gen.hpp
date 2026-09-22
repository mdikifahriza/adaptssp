#pragma once
// FASE 2 - Generate kandidat per blok via DFS + suffix pruning.
//
// Berbeda dari Gosper's hack (yang HANYA membangkitkan subset dengan bobot
// PERSIS w), fungsi ini membangkitkan SEMUA subset dari `block` yang
// bobotnya jatuh di window [w_lo, w_hi] (inklusif). Ini penting karena
// solusi asli belum tentu split rata w=m/4 persis di tiap blok.
//
// Pruning yang dipakai (keduanya provably sound, tidak membuang kandidat
// valid):
//   1) weight > w_hi  -> subtree ini sudah pasti melebihi batas atas.
//   2) (sisa elemen yang belum diputuskan) < (w_lo - weight) -> walaupun
//      SEMUA elemen sisa diambil, bobot tidak akan pernah mencapai w_lo.
#include <vector>
#include <cstdint>
#include "common.hpp"

struct BlockCandidate {
    int128 sum;
    uint32_t local_mask; // bit j set berarti block[j] terpakai (j relatif ke awal blok)
};

// DFS implementasi sebagai fungsi bebas (bukan std::function/lambda capture)
// supaya compiler bisa inline penuh -- untuk block_size=24 ini dipanggil
// puluhan juta kali per blok, jadi overhead pemanggilan virtual/indirect
// dari std::function terasa signifikan (>10x lebih lambat dari versi ini).
struct BlockGenCtx {
    const int128* vals;
    int n;
    int w_lo, w_hi;
    int128 sum_stack[65];
    uint32_t mask_stack[65];
    std::vector<BlockCandidate>* out;
};

inline void dfs_gen(BlockGenCtx& ctx, int i, int weight) {
    if (weight > ctx.w_hi) return;
    if ((ctx.n - i) < (ctx.w_lo - weight)) return;
    if (i == ctx.n) {
        if (weight >= ctx.w_lo && weight <= ctx.w_hi) {
            ctx.out->push_back({ctx.sum_stack[i], ctx.mask_stack[i]});
        }
        return;
    }
    // exclude vals[i]
    ctx.sum_stack[i + 1] = ctx.sum_stack[i];
    ctx.mask_stack[i + 1] = ctx.mask_stack[i];
    dfs_gen(ctx, i + 1, weight);
    // include vals[i]
    ctx.sum_stack[i + 1] = ctx.sum_stack[i] + ctx.vals[i];
    ctx.mask_stack[i + 1] = ctx.mask_stack[i] | (1u << i);
    dfs_gen(ctx, i + 1, weight + 1);
}

inline void gen_block_candidates(const std::vector<int128>& block,
                                  int w_lo, int w_hi,
                                  std::vector<BlockCandidate>& out) {
    int n = (int)block.size();
    w_lo = std::max(0, w_lo);
    w_hi = std::min(n, w_hi);
    if (w_lo > w_hi) return;
    if (n > 64) return; // guard untuk sum_stack/mask_stack fixed-size

    double est = 0.0;
    for (int w = w_lo; w <= w_hi; ++w) est += n_choose_k_d(n, w);
    if (est > 0 && est < 5e8) out.reserve(out.size() + (size_t)est + 16);

    BlockGenCtx ctx;
    ctx.vals = block.data();
    ctx.n = n;
    ctx.w_lo = w_lo;
    ctx.w_hi = w_hi;
    ctx.sum_stack[0] = 0;
    ctx.mask_stack[0] = 0;
    ctx.out = &out;
    dfs_gen(ctx, 0, 0);
}
