#!/usr/bin/env python3
"""Tes diferensial solver SS (exhaustive, --k_radius -1) vs brute force MITM.

Pemakaian:  difftest.py BINARY COUNT SEED [--big] [--keep]

Tiap instance: tulis .prb sementara, jalankan `BINARY -f X.prb --solver ss`, lalu bandingkan:
  - oracle bilang ada solusi tapi solver tidak menemukan  -> MISS
  - solver bilang ketemu tapi oracle bilang tidak ada     -> FALSE_POSITIVE
  - solver ketemu tapi .sol tidak berjumlah target        -> BAD_SOL
  - proses crash/timeout (rc != 0)                        -> CRASH (rc dicetak)
Mode biasa: n 4..22, nilai kecil/duplikat/nol, target = jumlah subset acak, acak, 0, atau total.
Mode --big : n=16, nilai 2^61..2^62.3, target = jumlah 3 elemen (total > 2^64 -> uji Bug 2).
Kode keluar 1 bila ada kegagalan.
"""
import os
import random
import subprocess
import sys
import tempfile


def oracle_feasible(vals, target):
    n = len(vals)
    h = n // 2
    L = [0]
    for v in vals[:h]:
        L += [x + v for x in L]
    R = [0]
    for v in vals[h:]:
        R += [x + v for x in R]
    rs = set(R)
    return any((target - ls) in rs for ls in L)


def gen(rng, big):
    if big:
        n = 16
        lo, hi = 1 << 61, int((1 << 62) * 1.23)
        vals = [rng.randint(lo, hi) for _ in range(n)]
        target = sum(rng.sample(vals, 3))
        return vals, target
    n = rng.randint(4, 22)
    mode = rng.randint(0, 3)
    if mode == 0:
        vals = [rng.randint(0, 5) for _ in range(n)]
    elif mode == 1:
        vals = [rng.randint(1, 12) for _ in range(n)]
    elif mode == 2:
        vals = [rng.randint(1, 1 << 20) for _ in range(n)]
    else:
        pool = [rng.randint(1, 1 << 30) for _ in range(4)]
        vals = [rng.choice(pool) for _ in range(n)]
    t = rng.randint(0, 3)
    if t == 0:
        target = sum(v for v in vals if rng.random() < 0.5)
    elif t == 1:
        target = rng.randint(0, max(1, sum(vals)))
    elif t == 2:
        target = 0
    else:
        target = sum(vals)
    return vals, target


def main():
    a = sys.argv[1:]
    if len(a) < 3:
        print(__doc__); sys.exit(2)
    binary, count, seed = os.path.abspath(a[0]), int(a[1]), int(a[2])
    big = "--big" in a
    rng = random.Random(seed)
    stats = {"MISS": 0, "FALSE_POSITIVE": 0, "BAD_SOL": 0, "CRASH": 0}
    feas = 0
    with tempfile.TemporaryDirectory() as tmp:
        for it in range(count):
            vals, target = gen(rng, big)
            name = "t%d.prb" % it
            with open(os.path.join(tmp, name), "w") as f:
                f.write("1 %d\n%s %d\n" % (len(vals), " ".join(map(str, vals)), target))
            exp = oracle_feasible(vals, target)
            feas += exp
            try:
                r = subprocess.run([binary, "-f", name, "--solver", "ss", "--k_radius", "-1"],
                                   cwd=tmp, capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired:
                stats["CRASH"] += 1
                print("[%d] TIMEOUT n=%d target=%d" % (it, len(vals), target))
                continue
            out = r.stdout
            got = "Found feasible solution!" in out
            problem = None
            if r.returncode != 0:
                problem = "CRASH"
            elif exp and not got:
                problem = "MISS"
            elif got and not exp:
                problem = "FALSE_POSITIVE"
            elif got:
                sp = os.path.join(tmp, "t%d.sol" % it)
                try:
                    bits = open(sp).read().split()[0]
                    s = sum(v for v, c in zip(vals, bits) if c == "1")
                    if len(bits) != len(vals) or s != target:
                        problem = "BAD_SOL"
                except Exception:
                    problem = "BAD_SOL"
            if problem:
                stats[problem] += 1
                print("[%d] %s rc=%d n=%d target=%d vals=%s" % (it, problem, r.returncode, len(vals), target, vals))
    bad = sum(stats.values())
    print("difftest %s: %d instance (%d feasible), gagal=%d %s" %
          ("--big" if big else "", count, feas, bad, stats))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
