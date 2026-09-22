#!/usr/bin/env python3
"""Brute-force eksak (MITM, bilangan bulat Python) untuk instance .prb kecil (N <= 30).

Pemakaian:
  bruteforce.py INST.prb [--limit L]
      -> cetak feasible ya/tidak dan jumlah solusi (berhenti mengumpulkan di L, default 200000).
  bruteforce.py INST.prb --component SOLFILE [--m 4] [--extsol FILE [--max-solutions M]]
      -> hitung komponen terhubung dari solusi di SOLFILE lewat swap <= m per sisi (oracle BFS,
         definisi sama dengan zero_sum_swap.h: |S\\T|<=m dan |T\\S|<=m), lalu bandingkan dengan FILE.
         Tanpa --max-solutions (atau bila jumlah baris < M): himpunan HARUS sama persis.
         Bila jumlah baris == M (kena cap): FILE harus himpunan bagian dari komponen.
Kode keluar 0 = lulus / informasi saja, 1 = beda, 3 = tidak bisa diputuskan (terlalu besar).
"""
import sys


def load_prb(path):
    tok = [int(x) for x in open(path).read().split()]
    if len(tok) < 4 or tok[0] != 1:
        raise SystemExit("bukan .prb 1D")
    n = tok[1]
    return tok[2:2 + n], tok[2 + n]


def all_solutions(vals, target, limit):
    n = len(vals)
    h = n // 2
    L = [0]
    for v in vals[:h]:
        L += [x + v for x in L]
    R = [0]
    for v in vals[h:]:
        R += [x + v for x in R]
    d = {}
    for m, s in enumerate(R):
        d.setdefault(s, []).append(m)
    sols = []
    truncated = False
    for lm, ls in enumerate(L):
        for rm in d.get(target - ls, ()):
            sols.append(lm | (rm << h))
            if len(sols) >= limit:
                truncated = True
                return sols, truncated
    return sols, truncated


def popcount(x):
    return bin(x).count("1")


def component(sols, start, m):
    """BFS pada graf solusi. Mengembalikan (himpunan_komponen, ukuran_lapis)."""
    remaining = set(sols)
    if start not in remaining:
        raise SystemExit("solusi awal bukan solusi instance ini")
    remaining.discard(start)
    layers = [1]
    frontier = [start]
    comp = {start}
    while frontier:
        nxt = []
        for S in frontier:
            hit = [T for T in remaining if popcount(S & ~T) <= m and popcount(T & ~S) <= m]
            for T in hit:
                remaining.discard(T)
                nxt.append(T)
        if nxt:
            layers.append(len(nxt))
            comp.update(nxt)
        frontier = nxt
    return comp, layers


def to_mask(line):
    x = 0
    for i, c in enumerate(line):
        if c == "1":
            x |= 1 << i
    return x


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__); sys.exit(2)
    inst = a[0]

    def opt(name, default=None):
        return a[a.index(name) + 1] if name in a else default

    limit = int(opt("--limit", 200000))
    vals, target = load_prb(inst)
    if len(vals) > 30:
        print("N > 30: terlalu besar untuk brute force MITM Python"); sys.exit(3)
    sols, truncated = all_solutions(vals, target, limit)
    print("n=%d target=%d feasible=%s solusi=%d%s" %
          (len(vals), target, "ya" if sols else "tidak", len(sols), " (TERPOTONG di limit)" if truncated else ""))

    if "--component" not in a:
        return
    m = int(opt("--m", 4))
    solfile = opt("--component")
    start = to_mask([l.strip() for l in open(solfile) if l.strip()][0])
    if truncated:
        print("tidak bisa memutuskan: solusi terpotong; naikkan --limit"); sys.exit(3)
    if len(sols) > 6000:
        print("SKIP: %d solusi > 6000 (BFS O(N^2) di Python terlalu lambat); pakai instance dgn solusi lebih sedikit" % len(sols))
        sys.exit(3)
    comp, layers = component(sols, start, m)
    print("komponen dari %s (m=%d): %d solusi, ukuran lapis BFS = %s" % (solfile, m, len(comp), layers))

    extfile = opt("--extsol")
    if not extfile:
        return
    ext = [to_mask(l.strip()) for l in open(extfile) if l.strip()]
    ext_set = set(ext)
    if len(ext_set) != len(ext):
        print("FAIL: extsol berisi duplikat"); sys.exit(1)
    maxs = opt("--max-solutions")
    capped = maxs is not None and int(maxs) != 0 and len(ext) >= int(maxs)
    if capped:
        ok = ext_set <= comp
        print("%s: extsol kena cap (%d baris); subset dari komponen: %s" % ("OK" if ok else "FAIL", len(ext), ok))
        sys.exit(0 if ok else 1)
    if ext_set == comp:
        print("OK: extsol == komponen oracle (%d solusi)" % len(comp))
        return
    print("FAIL: extsol (%d) != komponen oracle (%d); hanya di extsol=%d, hanya di oracle=%d" %
          (len(ext_set), len(comp), len(ext_set - comp), len(comp - ext_set)))
    sys.exit(1)


if __name__ == "__main__":
    main()
