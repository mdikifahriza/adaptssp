#!/usr/bin/env python3
"""Verifikasi berkas solusi terhadap instance .prb (aritmetika bilangan bulat Python, eksak).

Pemakaian:
  verify_sol.py INSTANCE.prb [SOLFILE]            # default SOLFILE = <folder instance>/<nama instance>.sol
  verify_sol.py INSTANCE.prb --extsol [FILE]      # default FILE   = <folder instance>/<nama instance>.extsol

Mode .sol   : baris pertama harus bitstring sepanjang N dengan jumlah == target.
Mode extsol : SEMUA baris harus valid (panjang N, jumlah == target), unik, dan baris 1 harus
              sama dengan isi .sol bila .sol ada.
Kode keluar 0 = lulus, 1 = gagal.
"""
import os
import sys


def load_prb(path):
    with open(path) as f:
        tok = [int(x) for x in f.read().split()]
    if len(tok) < 4 or tok[0] != 1:
        raise SystemExit("bukan .prb 1D: %s" % path)
    n = tok[1]
    if len(tok) < 2 + n + 1:
        raise SystemExit("token kurang di %s" % path)
    return tok[2:2 + n], tok[2 + n]


def check_line(line, vals, target):
    if len(line) != len(vals) or any(c not in "01" for c in line):
        return "panjang/karakter salah (len=%d, n=%d)" % (len(line), len(vals))
    s = sum(v for v, c in zip(vals, line) if c == "1")
    if s != target:
        return "jumlah %d != target %d" % (s, target)
    return None


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__); sys.exit(2)
    inst = a[0]
    ext = "--extsol" in a
    rest = [x for x in a[1:] if x != "--extsol"]
    base = os.path.splitext(os.path.basename(inst))[0]
    inst_dir = os.path.dirname(inst) or "."
    vals, target = load_prb(inst)

    if not ext:
        path = rest[0] if rest else os.path.join(inst_dir, base + ".sol")
        lines = [l.strip() for l in open(path) if l.strip()]
        if not lines:
            print("FAIL: %s kosong" % path); sys.exit(1)
        err = check_line(lines[0], vals, target)
        if err:
            print("FAIL: %s: %s" % (path, err)); sys.exit(1)
        print("OK: %s valid (%d elemen terpilih)" % (path, lines[0].count("1")))
        return

    path = rest[0] if rest else os.path.join(inst_dir, base + ".extsol")
    lines = [l.strip() for l in open(path) if l.strip()]
    if not lines:
        print("FAIL: %s kosong" % path); sys.exit(1)
    bad = 0
    for i, l in enumerate(lines, 1):
        err = check_line(l, vals, target)
        if err:
            print("FAIL baris %d: %s" % (i, err)); bad += 1
            if bad >= 10:
                break
    if len(set(lines)) != len(lines):
        print("FAIL: ada baris duplikat (%d unik dari %d)" % (len(set(lines)), len(lines))); bad += 1
    solp = os.path.join(os.path.dirname(path) or ".", base + ".sol")
    if os.path.exists(solp):
        first = [l.strip() for l in open(solp) if l.strip()]
        if first and first[0] != lines[0]:
            print("FAIL: baris 1 %s != isi %s" % (path, solp)); bad += 1
    if bad:
        sys.exit(1)
    print("OK: %s: %d solusi, semua valid dan unik" % (path, len(lines)))


if __name__ == "__main__":
    main()
