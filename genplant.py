#!/usr/bin/env python3
"""Generator instance subset-sum 1D dengan solusi tertanam (planted), format .prb.

Pemakaian:
  genplant.py OUT.prb N BITS SEED [--k K] [--dups D]

  N     jumlah elemen
  BITS  nilai acak di [1, 2^BITS)   (BITS=63 dan N=32 -> total > 2^64, memicu jalur u128)
  SEED  seed RNG
  --k K       ukuran solusi tertanam (default N//2)
  --dups D    ambil nilai dari kolam D nilai berbeda (banyak duplikat -> banyak solusi;
              berguna untuk menguji --extsol)

Format .prb:  "1 N\n v1 v2 ... vN TARGET\n".  Indeks tertanam dicetak ke stdout.
"""
import random
import sys


def main():
    args = sys.argv[1:]
    if len(args) < 4:
        print(__doc__)
        sys.exit(2)
    out, n, bits, seed = args[0], int(args[1]), int(args[2]), int(args[3])
    k = n // 2
    dups = 0
    i = 4
    while i < len(args):
        if args[i] == "--k":
            k = int(args[i + 1]); i += 2
        elif args[i] == "--dups":
            dups = int(args[i + 1]); i += 2
        else:
            print("argumen tidak dikenal:", args[i]); sys.exit(2)
    if not (1 <= k <= n):
        print("K harus di 1..N"); sys.exit(2)

    rng = random.Random(seed)
    hi = (1 << bits) - 1
    if dups > 0:
        pool = [rng.randint(1, hi) for _ in range(dups)]
        vals = [rng.choice(pool) for _ in range(n)]
    else:
        vals = [rng.randint(1, hi) for _ in range(n)]
    planted = sorted(rng.sample(range(n), k))
    target = sum(vals[j] for j in planted)

    with open(out, "w") as f:
        f.write("1 %d\n" % n)
        f.write(" ".join(str(v) for v in vals) + " %d\n" % target)
    print("wrote %s: n=%d bits=%d k=%d total=%d (2^%.1f) target=%d" %
          (out, n, bits, k, sum(vals), max(sum(vals), 1).bit_length() - 1, target))
    print("planted indices:", planted)


if __name__ == "__main__":
    main()
