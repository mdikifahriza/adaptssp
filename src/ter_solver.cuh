#pragma once

#include <cstdint>
#include <vector>
#include <string>

using ter_u128 = unsigned __int128;

struct TerEntry {
    uint64_t psum;
    uint64_t pos_lo;
    uint64_t pos_hi;
    uint64_t neg_lo;
    uint64_t neg_hi;
};

struct Level1Sidecar {
    std::vector<uint64_t> a_ps;
    std::vector<uint64_t> b_ps;
    std::vector<uint64_t> c_keys;
    std::vector<uint32_t> start;
    int  shift = 0;
    bool valid = false;
};

struct TerParams {
    int    n;

    ter_u128 total_weight_sum = 0;

    double eps11;
    double eps01;
    double eps12;
    double eps02;
    double eps22;

    double w1;
    double w2;
    double l1, l2, l3;
    double r1, r2;
    int    b1, b2, b3;
    size_t target_L3, target_L2, target_L1;

    int    b2_delta = 0;

    int    max_restarts;
    double timeout_seconds;
    int    fixed_runs;
    bool   verbose;

    double heartbeat_seconds = 0.0;

    bool   use_bucket_lookup;
    bool   continue_after_found;

    void compute_derived();
};

struct TerResult {
    bool found;
    std::vector<size_t> solution_indices;
    int    runs_attempted;
    double elapsed_seconds;
    int    successful_runs;
};

TerParams ter_default_params(const std::vector<ter_u128>& weights);

void ter_set_cpu_split_frac_override(double frac);

TerResult ter_solve(
    const std::vector<ter_u128>& weights,
    ter_u128                     target,
    const TerParams&             params
);