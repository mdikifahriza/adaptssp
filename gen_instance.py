import random, sys

def gen_instance(n, seed=None, density=1.0):
    """
    Generate random subset-sum instance dgn karakteristik mirip benchmark HGJ:
    - n bilangan acak sekitar n-bit (bit length ~ n, sesuai density=1)
    - k = n//2 elemen dipilih acak sbg solusi planted
    - target T = jumlah elemen yg dipilih
    """
    if seed is not None:
        random.seed(seed)
    bitlen = int(n / density)
    lo = 1 << (bitlen - 1)
    hi = (1 << bitlen) - 1
    a = [random.randint(lo, hi) for _ in range(n)]
    k = n // 2
    idx = sorted(random.sample(range(n), k))
    T = sum(a[i] for i in idx)
    return a, T, idx, k

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    a, T, idx, k = gen_instance(n, seed=seed)
    print(f"# HGJ Benchmark Instance (N={n}, BitLength={n}, k={k}, seed={seed})")
    print(f"Target T: {T}")
    for v in a:
        print(f"    {v},")
    print()
    print(f"# planted solution indices (0-based): {idx}")
