#!/usr/bin/env bash
# Inisialisasi git + commit "titik aman" (rencana.md §12 poin 6).
#   bash tools/setup_git.sh                 # git init + vendor argparse + commit
#   REMOVE_OLD=1 bash tools/setup_git.sh    # sekalian hapus cuda_kernels.*, markshare.hpp (keputusan K1/§6)
set -eu
cd "$(dirname "$0")/.."

if [ "${REMOVE_OLD:-0}" = "1" ]; then
  for f in cuda_kernels.cu cuda_kernels.cuh markshare.hpp src/cuda_kernels.cu src/cuda_kernels.cuh src/markshare.hpp; do
    [ -f "$f" ] && { rm -f "$f"; echo "dihapus: $f"; }
  done
fi

[ -d .git ] || git init
if [ -f external/argparse/include/argparse/argparse.hpp ]; then
  # Vendor: buang .git milik clone argparse supaya isinya ikut ter-commit (tidak hilang di Colab baru).
  rm -rf external/argparse/.git
else
  git clone --depth 1 https://github.com/p-ranav/argparse.git external/argparse
  rm -rf external/argparse/.git
fi
git add -A
git -c user.name="${GIT_AUTHOR_NAME:-dev}" -c user.email="${GIT_AUTHOR_EMAIL:-dev@example.com}" \
    commit -m "${1:-titik aman}" || echo "(tidak ada perubahan untuk di-commit)"
git log --oneline | head -5
