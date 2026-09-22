#pragma once
#include <string>
#include <algorithm>
#include <cstdint>

typedef __int128_t int128;
typedef unsigned __int128 uint128;

inline int128 parse_int128(const std::string& s) {
    int128 res = 0;
    bool neg = false;
    size_t i = 0;
    while (i < s.length() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n')) i++;
    if (i < s.length() && s[i] == '-') { neg = true; i++; }
    for (; i < s.length(); i++) {
        if (s[i] >= '0' && s[i] <= '9') {
            res = res * 10 + (s[i] - '0');
        }
    }
    return neg ? -res : res;
}

inline std::string int128_to_string(int128 n) {
    if (n == 0) return "0";
    std::string s;
    bool neg = n < 0;
    if (neg) n = -n;
    while (n > 0) {
        s += (char)('0' + (int)(n % 10));
        n /= 10;
    }
    if (neg) s += '-';
    std::reverse(s.begin(), s.end());
    return s;
}

// n choose k as a double to avoid overflow when values get large;
// callers that need exact small counts can round.
inline double n_choose_k_d(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    if (k > n - k) k = n - k;
    double res = 1.0;
    for (int i = 0; i < k; ++i) {
        res = res * (double)(n - i) / (double)(i + 1);
    }
    return res;
}

inline uint64_t n_choose_k_u64(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    uint64_t res = 1;
    for (int i = 0; i < k; ++i) {
        res = res * (uint64_t)(n - i) / (uint64_t)(i + 1);
    }
    return res;
}
