#pragma once
// ter_solver.cuh — TER (Ternary Enumeration Representation) SSP Solver
// Based on: Yang Li, "From Subset-Sum to Decoding: Improved Classical and
//           Quantum Algorithms via Ternary Representation Technique"
//           Information 2025, 16(10), 887. DOI:10.3390/info16100887
//
// Algorithm 2 (TER depth-3): solves 1D Subset Sum in O~(2^{0.240n})
// time and O~(2^{0.222n}) space.
//
// Structure (18 L3 → 9 L2 → 3 L1 → 1 L0/root):
//   Level 3: 18 base lists (9 pairs, binary MITM split of n)
//   Level 2: 9 lists (merge L3 pairs via MITM on partial sum)
//   Level 1: 3 lists (merge L2 triplets — TERNARY merge, CPU)
//   Root:    1 (merge L1 triplet — TERNARY merge, check target)
//
// GPU acceleration: Level 1 merge (sort + binary-search on GPU)

#include <cstdint>
#include <vector>
#include <string>

// ─────────────────────────────────────────────────────────
// 128-bit weight/target type. Weights and target are stored at full
// precision so instances whose values or partial sums exceed 2^64
// (e.g. instance_n64.prb) are solved correctly instead of silently
// truncating/overflowing.
// ─────────────────────────────────────────────────────────
using ter_u128 = unsigned __int128;

// ─────────────────────────────────────────────────────────
// Entry in a ternary list.
// Represents a vector v ∈ {-1,0,+1}^n where n ≤ 128.
// For n=96 we need 96 bits → 2×uint64.
//
// NOTE on `psum`: this field is intentionally kept as a 64-bit value
// even though weights/target are now ter_u128 (128-bit). It is only
// ever used as a hash/residue: every matching step in the solver
// (Level2, Level1, and the candidate pre-filter at the root) inspects
// psum modulo 2^b for some b<=58, or compares psum for 64-bit
// equality as a cheap filter. Truncation to the low 64 bits commutes
// with addition/subtraction mod 2^64, so summing already-truncated
// 64-bit values yields exactly the correct low 64 bits of the true
// (arbitrary-width) sum — no precision is lost for matching. The one
// place that needs the *exact* wide sum — final candidate
// verification — is done by re-summing the original ter_u128 weights
// selected by the entry's bit pattern (pos_lo/pos_hi/neg_lo/neg_hi),
// which is exact regardless of psum's truncation.
//
// This keeps TerEntry at 40 bytes and every hot loop (including the
// GPU kernel) operating on plain 64-bit integers, so there is no
// performance cost for instances that fit in 64 bits, while instances
// that don't are still solved correctly via the wide re-verification.
// ─────────────────────────────────────────────────────────
struct TerEntry {
    uint64_t psum;      // low 64 bits of partial sum <a, v> — hash/residue only, see note above.
                         // Unsigned so that the additions performed on it in
                         // merge_level2/merge_level1/merge_root_and_solve (and
                         // the GPU kernel) are well-defined wraparound instead
                         // of signed-overflow UB (Bug 1, see rencana.md §4.1).
    uint64_t pos_lo;    // bits 0..63:  pos_lo[i]=1 ↔ v[i]=+1
    uint64_t pos_hi;    // bits 64..95: pos_hi[i]=1 ↔ v[i+64]=+1
    uint64_t neg_lo;    // bits 0..63:  neg_lo[i]=1 ↔ v[i]=-1
    uint64_t neg_hi;    // bits 64..95: neg_hi[i]=1 ↔ v[i+64]=-1
};
// 40 bytes per entry.
// Constraint: pos_lo & neg_lo == 0, pos_hi & neg_hi == 0

// ─────────────────────────────────────────────────────────
// Sidecar untuk merge_level1 (rencana.md §5.1 poin 10, langkah 4c).
// Array kunci uint64 terpisah + bucket table di atas `sorted_C`, supaya
// pencarian pasangan tidak menyentuh TerEntry 40 B kecuali ada kecocokan
// kunci. Dibangun ulang tiap panggilan merge_level1 (O(|A|+|B|+|C|)).
//   c_keys[k] = sorted_C[k].psum & mask_m1        (harus terurut naik)
//   start[b]  = indeks pertama k dengan (c_keys[k] >> shift) >= b
//   bucket(req) = req >> shift, shift = b1 - bits, bits = floor(log2 |C|)
// Hasil pencarian identik dengan binary search penuh (hanya lebih murah).
// ─────────────────────────────────────────────────────────
struct Level1Sidecar {
    std::vector<uint64_t> a_ps;    // psum tiap entri A (urutan sama dengan A)
    std::vector<uint64_t> b_ps;    // psum tiap entri B (urutan sama dengan B)
    std::vector<uint64_t> c_keys;  // (psum & mask_m1) tiap entri sorted_C
    std::vector<uint32_t> start;   // 2^bits + 1 offset bucket
    int  shift = 0;
    bool valid = false;
};

// ─────────────────────────────────────────────────────────
// Optimal TER parameters for a given n.
// All values from numerical optimization (ter_run.py).
// ─────────────────────────────────────────────────────────
struct TerParams {
    int    n;          // instance size

    // Total magnitude of the actual instance weights (sum of all |w_i|,
    // full 128-bit precision). Set by ter_default_params() from the real
    // weights BEFORE compute_derived() runs. This is what makes the
    // b1/b2 clamp below adaptive to the instance itself rather than to a
    // hardcoded n range: two instances with the same n but very
    // different weight magnitudes get different, correctly-sized clamps.
    // Zero means "unknown" (e.g. legacy callers) and disables the clamp.
    ter_u128 total_weight_sum = 0;

    double eps11;      // ε(1)_1 : frac 1-coords with rep type 1+1+(-1) at level 1
    double eps01;      // ε(1)_0 : frac 0-coords with non-trivial rep at level 1
    double eps12;      // ε(2)_1 : frac 1-coords with rep type 1+1+(-1) at level 2
    double eps02;      // ε(2)_0 : frac 0-coords with non-trivial rep at level 2
    double eps22;      // ε(2)_{-1}: frac (-1)-coords with rep type 1+(-1)+(-1) at level 2

    // Derived automatically from eps values:
    double w1;         // frac of -1 entries in e1,e2,e3 at level 1
    double w2;         // frac of -1 entries at level 2
    double l1, l2, l3; // log2(list size) / n per level
    double r1, r2;     // log2(representations) / n per level
    int    b1, b2, b3; // bits matched at level 1, 2, 3 merge
    size_t target_L3, target_L2, target_L1; // actual target list sizes per level
                                              // (floored versions of 2^(l*n); see compute_derived())

    // Solver behavior
    int    max_restarts;      // 0 = infinite until timeout
    double timeout_seconds;   // 0 = no timeout
    int    fixed_runs;        // if > 0, run exactly this many times (overrides max_restarts)
    bool   verbose;

    // Langkah 4c: pakai sidecar key + bucket lookup di merge_level1 (CPU dan GPU)
    // menggantikan binary search. Default false = jalur lama (perilaku tidak berubah).
    bool   use_bucket_lookup;
    // Langkah 4b: jangan berhenti di solusi pertama; lanjut sampai fixed_runs /
    // max_restarts / timeout habis dan hitung success rate per run. Diabaikan bila
    // tidak ada satu pun batas run yang diset (supaya tidak berjalan tanpa akhir).
    bool   continue_after_found;

    // Reconstruct derived params from eps values
    void compute_derived();
};

// ─────────────────────────────────────────────────────────
// Result of one TER run
// ─────────────────────────────────────────────────────────
struct TerResult {
    bool found;
    std::vector<size_t> solution_indices;  // 0-indexed positions in weights[]
    int    runs_attempted;
    double elapsed_seconds;
    int    successful_runs;                // run yang menemukan solusi (>1 hanya bila continue_after_found)
};

// ─────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────

// Returns optimal params for the given instance. Takes the actual weights
// (not just n) so total_weight_sum can be filled in and used by
// compute_derived() to adaptively clamp b1/b2 to the instance's real
// magnitude — see the comment on TerParams::total_weight_sum and on
// compute_derived() for why this matters.
TerParams ter_default_params(const std::vector<ter_u128>& weights);

// Main solver entry point.
// weights/target are full 128-bit precision (ter_u128) so instances
// with values or partial sums beyond 2^64 are handled correctly.
// Internally, instances that actually fit in 64 bits still take the
// exact same 64-bit hot path as before (see TerEntry note above) —
// there is no performance penalty for small instances.
TerResult ter_solve(
    const std::vector<ter_u128>& weights,
    ter_u128                     target,
    const TerParams&             params
);
