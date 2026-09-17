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
// Entry in a ternary list.
// Represents a vector v ∈ {-1,0,+1}^n where n ≤ 128.
// For n=96 we need 96 bits → 2×uint64.
// ─────────────────────────────────────────────────────────
struct TerEntry {
    int64_t  psum;      // partial sum  = <a, v>  (signed, mod 2^64)
    uint64_t pos_lo;    // bits 0..63:  pos_lo[i]=1 ↔ v[i]=+1
    uint64_t pos_hi;    // bits 64..95: pos_hi[i]=1 ↔ v[i+64]=+1
    uint64_t neg_lo;    // bits 0..63:  neg_lo[i]=1 ↔ v[i]=-1
    uint64_t neg_hi;    // bits 64..95: neg_hi[i]=1 ↔ v[i+64]=-1
};
// 40 bytes per entry.
// Constraint: pos_lo & neg_lo == 0, pos_hi & neg_hi == 0

// ─────────────────────────────────────────────────────────
// Optimal TER parameters for a given n.
// All values from numerical optimization (ter_run.py).
// ─────────────────────────────────────────────────────────
struct TerParams {
    int    n;          // instance size
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

    // Solver behavior
    int    max_restarts;      // 0 = infinite until timeout
    double timeout_seconds;   // 0 = no timeout
    int    fixed_runs;        // if > 0, run exactly this many times (overrides max_restarts)
    bool   verbose;
    bool   use_gpu;

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
};

// ─────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────

// Returns optimal params for given n (uses precomputed or runs optimizer)
TerParams ter_default_params(int n);

// Main solver entry point
TerResult ter_solve(
    const std::vector<uint64_t>& weights,
    uint64_t                     target,
    const TerParams&             params
);
