#pragma once

#include <immintrin.h>
#include <smmintrin.h> // SSE4.1
#include <nmmintrin.h> // SSE4.2
#include <wmmintrin.h> // AES-NI
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>

using u64  = uint64_t;
using u32  = uint32_t;
using u128 = unsigned __int128;

// ============================================================================
// 1. Bitmask & PTEST Disjoint Checker (SSE4.1)
// ============================================================================

// PTEST mengevaluasi (a & b) == 0 dalam 1 siklus clock tanpa conditional jump
inline bool is_disjoint_u64(u64 a, u64 b) {
    __m128i va = _mm_cvtsi64_si128((long long)a);
    __m128i vb = _mm_cvtsi64_si128((long long)b);
    return _mm_testz_si128(va, vb) != 0;
}

inline bool is_disjoint_m128(__m128i a, __m128i b) {
    return _mm_testz_si128(a, b) != 0;
}

// ============================================================================
// 2. Hardware CRC32 Hashing (SSE4.2)
// ============================================================================

// Menghitung hash kunci 64-bit pada level sirkuit silikon (latensi 3 siklus)
inline u64 crc32_fast_hash(u64 key, u64 seed = 0x517cc1b727220a95ULL) {
    return _mm_crc32_u64(seed, key);
}

// ============================================================================
// 3. Counter-based AES-NI Pseudo Random Number Generator
// ============================================================================

// Pembangkit bilangan acak berbasis AES-NI round encryption.
// Jauh lebih cepat dan merata daripada std::mt19937_64.
struct AesPrng {
    __m128i round_key;
    u64 counter;

    AesPrng(u64 seed = 13371337ULL) {
        counter = seed;
        round_key = _mm_set_epi64x((long long)0x9E3779B97F4A7C15ULL, (long long)0xBF58476D1CE4E5B9ULL);
    }

    inline u64 next_u64() {
        __m128i c = _mm_set_epi64x((long long)(++counter), (long long)(counter ^ 0x5555555555555555ULL));
        __m128i enc = _mm_aesenc_si128(c, round_key);
        return (u64)_mm_cvtsi128_si64(enc);
    }

    inline u64 next_range(u64 limit) {
        if (limit <= 1) return 0;
        // Lemire fast alternative to modulo
        u64 x = next_u64();
        u128 m = (u128)x * (u128)limit;
        return (u64)(m >> 64);
    }

    inline u128 next_u128() {
        u64 lo = next_u64();
        u64 hi = next_u64();
        return ((u128)hi << 64) | lo;
    }
};

// ============================================================================
// 4. Branchless 4-Way SIMD Modulo Addition (SSE2 / SSE4.1)
// ============================================================================

// Menjumlahkan 4 elemen residu 32-bit modulo M secara paralel tanpa percabangan
inline __m128i add_mod4_simd(__m128i a, __m128i b, __m128i M_vec) {
    __m128i sum = _mm_add_epi32(a, b);
    __m128i mask = _mm_cmpgt_epi32(sum, _mm_sub_epi32(M_vec, _mm_set1_epi32(1)));
    return _mm_sub_epi32(sum, _mm_and_si128(mask, M_vec));
}

// ============================================================================
// 5. Vectorized Digit-Sum Validation (SSE4.1) untuk 16 Koordinat Sekaligus
// ============================================================================

// Memeriksa apakah penjumlahan dua vektor int8 berpanjang 16 berada dalam rentang [lo, hi]
inline bool try_sum_16bytes(const int8_t* x, const int8_t* y, int8_t lo, int8_t hi, int8_t* out) {
    __m128i vx = _mm_loadu_si128((const __m128i*)x);
    __m128i vy = _mm_loadu_si128((const __m128i*)y);
    __m128i sum = _mm_add_epi8(vx, vy);

    __m128i vlo = _mm_set1_epi8(lo);
    __m128i vhi = _mm_set1_epi8(hi);

    // Cek jika ada yang < lo atau > hi
    __m128i below = _mm_cmplt_epi8(sum, vlo);
    __m128i above = _mm_cmpgt_epi8(sum, vhi);
    __m128i invalid = _mm_or_si128(below, above);

    // Jika invalid == 0 (semua koordinat sah)
    if (_mm_testz_si128(invalid, invalid)) {
        _mm_storeu_si128((__m128i*)out, sum);
        return true;
    }
    return false;
}

// ============================================================================
// 6. Flat Compact Hash Map khusus 64-bit dengan Hardware CRC32 (SSE4.2)
// ============================================================================
struct CompactHashMap {
    struct Slot {
        u64 key;
        u32 index;
        u32 next;
    };
    std::vector<u32> head;
    std::vector<Slot> slots;
    u64 mask_cap = 0;

    void init(size_t expected_elements) {
        size_t cap = 1;
        while (cap < expected_elements * 2) cap <<= 1;
        if (cap < 1024) cap = 1024;
        mask_cap = cap - 1;
        head.assign(cap, 0xFFFFFFFFU);
        slots.clear();
        slots.reserve(expected_elements);
    }

    inline void insert(u64 key, u32 idx) {
        u64 bucket = crc32_fast_hash(key) & mask_cap;
        slots.push_back({key, idx, head[bucket]});
        head[bucket] = (u32)(slots.size() - 1);
    }
};

