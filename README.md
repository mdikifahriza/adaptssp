# adaptssp

`adaptssp` is a command-line application for solving the **1-dimensional subset sum** problem: given a set of values and a target, find a subset whose sum equals the target.

This repository provides two search engines:

- **TER (Ternary Enumeration Representation)** as the default solver, with an OpenMP CPU pipeline and a CUDA GPU pipeline.
- **Schroeppel–Shamir** as an alternative that splits the input into four parts and uses a heap merge.

Once the first solution is found, the application can explore other solutions connected through **zero-sum swaps**.

## Quick start on Google Colab (recommended)

`runcollab.ipynb` is the working example for running `adaptssp` on Google Colab. It is a ready-to-run walkthrough, not a log of old experiments: it clones the source, builds `build/adaptssp` with CUDA, and solves the bundled instances on a Colab T4 GPU.

To try it:

1. Download `runcollab.ipynb` from the repository root.
2. Open [Google Colab](https://colab.research.google.com/), then use **File → Upload notebook** and select `runcollab.ipynb`.
3. In the Colab menu, choose **Runtime → Change runtime type**, set **Hardware accelerator** to **T4 GPU**, then click **Save**.
5. Run all cells

You can also open the ready-made notebook directly:

- https://colab.research.google.com/drive/1S9O0fSsP6A4MRSmhO5bz7EITs2zMLtar?usp=sharing

### What the notebook does

| Step | Cell | Action |
|---|---|---|
| 1 | *Connect google collab with t4 then Donwload source code* | Clones the repository, copies its contents into `/content`, and removes the leftover clone directory. |
| 2 | *Clone argparse* | Fetches the `argparse` dependency into `external/argparse`. |
| 3 | *Build* | Configures CMake in Release mode, builds with `cmake --build build -j2`, producing `build/adaptssp`. |
| 4 | *(no heading)* | Runs the TER solver on `instance_n16`, `instance_n24`, `instance_n32`, and `instance_n40` with `--solver ter --autorestart`. |
| 5 | *This may take longer* | Runs the same command on `instance_n48` and `instance_n56`; `--autorestart` keeps retrying until a solution is found. |
| 6 | *download all soll files* | Finds every `*.sol` file, packs them into `sol_files.zip`, and downloads it. |


## Running from the CLI

For command-line usage and the full option reference, see [`cli.md`](cli.md). The notebook in this repository is the fastest path; the CLI is the way to script repeated runs, tune solver parameters, and collect statistics.

Common invocations:

```bash
# Inspect an instance without running a solver
./build/adaptssp work/instance_n24.prb --check_only

# Default TER solver, one run
./build/adaptssp work/instance_n24.prb

# Keep restarting TER until a solution appears, bounded at 60 seconds
./build/adaptssp work/instance_n24.prb --solver ter --autorestart --timeout 60

# 20 runs plus per-phase statistics
./build/adaptssp work/instance_n24.prb --solver ter --runs 20 --ter_stats

# Schroeppel–Shamir over the whole cardinality window
./build/adaptssp work/instance_n64.prb --solver ss --k_radius -1

# Schroeppel–Shamir, priority-peak cardinality only, GPU quarter sorting
./build/adaptssp work/instance_n96.prb --solver ss --k_radius 0 --ss_gpu

# Explore connected solutions after the first one
./build/adaptssp work/instance_n24.prb --extsol
```

## Features

- `.prb` instance parser with values and target as unsigned 128-bit (`u128`).
- Hybrid CPU/GPU TER solver with adaptive work splitting and CPU fallback.
- Schroeppel–Shamir solver with a cardinality range heuristic.
- Optional GPU quarter sorting for Schroeppel–Shamir using CUB.
- `.extsol` exploration of solutions connected to the first solution.
- TER RNG uses the fixed seed `1337`, so the remainder order is reproducible on the same configuration.
- Fixture instances and a set of reference solutions in `work/`.

## Algorithms

### TER

TER splits the items into two halves and builds a hierarchical ternary representation:

1. Build the L3 base pool from positive/negative assignments on each half.
2. Join L2 using the remainder modulo `b2`.
3. Join L1 using the remainder modulo `b1`.
4. Join the root, drop all negative bits, then re-verify the sum with `u128`.
5. If a run finds no solution, the next run uses a fresh random remainder.

The L1 join can be split between CPU and GPU. The CPU side uses OpenMP plus a sidecar bucket, while the GPU side uses a CUDA kernel for matching. The CPU share can be overridden or balanced adaptively.

### Schroeppel–Shamir

The input is split into four quarters. The solver:

1. Computes the window of possible cardinalities.
2. Orders cardinalities by a priority heuristic.
3. Generates the subset sums of each quarter and filters them by cardinality.
4. Sorts two quarters as ascending/descending streams.
5. Searches for matching pairs with two k-way heap merges and verifies the final sum.

With `--k_radius 0`, only the priority-peak cardinality is tried. With `--k_radius -1`, the whole cardinality window is tried.

### Zero-sum swap

`--extsol` starts from an already verified solution. For each solution, the application separates the selected and unselected items, matches subsets with equal sums, then runs a tiered BFS to discover new solutions.

## Input format

An input file is a whitespace-separated text file with the format:

```text
1 n value_1 value_2 ... value_n target
```

- The first token must be `1`, marking a one-dimensional subset sum instance.
- The second token is the number of values `n`.
- Exactly `n` item values follow.
- The last token is the target.

A small example instance:

```text
1 4
3 5 7 11
10
```

The solution is `3 + 7 = 10`, with bitstring `1010` (0-based indices).

## CLI options

### General options

| Option | Default | Description |
|---|---:|---|
| `instance` | — | Path to the `.prb` file; required. |
| `-t`, `--threads N` | all logical cores | Number of OpenMP threads. |
| `--check_only` | off | Only parse and print instance metadata. |
| `--max_pairs N` | automatic | Batch limit for pairs. If not given, it is computed from the memory budget; mainly used for Schroeppel–Shamir pair chunking. |
| `--mem_budget_gb G` | `0` | Memory budget. `0` means auto-detect RAM from `/proc/meminfo`, with an 8 GiB fallback. |

Automatic `--max_pairs` calculation uses about 60% of the memory budget and 64 bytes per pair, with a minimum of 1000 pairs. TER has its own L1/L2 capacity limits computed by the solver.

### TER options

| Option | Default | Description |
|---|---:|---|
| `--solver ter` | active | Selects TER. |
| `--runs N` | `0` | Number of attempts. Without `--autorestart`, `0` becomes a single run. |
| `--autorestart` | off | Keeps running TER until a solution is found. |
| `--timeout SECONDS` | `0` | Total time limit; `0` means no limit. Checked at run and stage boundaries. |
| `--ter_stats` | off | Runs exactly `--runs N` and prints per-phase statistics. |
| `--ter_cpu_frac F` | `-1` | Override the share of A rows processed on the CPU during the L1 join. `-1` uses adaptive balancing. |
| `--ter_b2_extra B` | `0` | Adds width to `b2`. |
| `--ter_heartbeat SECONDS` | `0` | Prints a `[TER-HB]` heartbeat; `0` disables it. |

`--ter_stats` requires `--runs N` with `N > 0` and cannot be combined with `--autorestart`.

### Schroeppel–Shamir options

| Option | Default | Description |
|---|---:|---|
| `--solver ss` | off | Selects Schroeppel–Shamir. |
| `--k_radius R` | `-1` | `-1` tries all cardinalities, `0` only the priority peak, and `R > 0` tries `peak_k ± R`. |
| `--ss_gpu` | off | Uses CUB to sort quarters larger than 65536 elements on the `uint64_t` path. The heap merge still runs on the CPU. |

### Zero-sum swap options

| Option | Default | Description |
|---|---:|---|
| `--extsol` | off | Runs zero-sum swap exploration after a solution is found. |
| `--extsol_swap_size S` | `4` | Maximum subset size per side, from 1 to 6. |
| `--extsol_max_solutions N` | `1000` | Limit on solutions including the initial one. `0` means no limit. |

## Output

When a solution is found, the application prints the selected indices (0-based) and saves a bitstring of `n` characters:

```text
1010
```

`.sol` and `.extsol` files are written relative to the **current working directory**, not the input file's directory. For example, the following command writes `instance_n24.sol` at the repository root:

```bash
./build/adaptssp work/instance_n24.prb
```

File formats:

- `<instance-name>.sol`: a single line holding the main solution bitstring.
- `<instance-name>.extsol`: one bitstring per solution, including the initial solution when exploration starts.

## Repository structure

```text
.
├── CMakeLists.txt
├── README.md
├── cli.md
├── runcollab.ipynb
├── src
│   ├── main.cpp
│   ├── pairs_tuple.hpp
│   ├── profiler.hpp
│   ├── ss_kernel.cu
│   ├── ss_kernel.cuh
│   ├── ter_kernel.cu
│   ├── ter_kernel.cuh
│   ├── ter_solver.cpp
│   ├── ter_solver.cuh
│   └── zero_sum_swap.h
└── work
    ├── instance_n16.prb
    ├── instance_n24.prb
    ├── instance_n32.prb
    ├── instance_n40.prb
    ├── instance_n48.prb
    ├── instance_n56.prb
    ├── instance_n64.prb
    ├── instance_n64_u64.prb
    ├── instance_n72.prb
    ├── instance_n80.prb
    ├── instance_n88.prb
    ├── instance_n96.prb
    └── instance_n96_u64.prb
```

### Main file roles

| File | Responsibility |
|---|---|
| `runcollab.ipynb` | Google Colab walkthrough: clone, fetch `argparse`, build, run TER on `instance_n16` … `instance_n56`, then download all `.sol` files. |
| `cli.md` | Full CLI reference and option behaviour. |
| `src/main.cpp` | CLI, `.prb` parser, thread/memory configuration, solver dispatch, Schroeppel–Shamir, solution output, and extension hooks. |
| `src/ter_solver.cuh` | TER data structures and public API. |
| `src/ter_solver.cpp` | Parameters, ternary pool, L2/L1/root joins, CPU OpenMP, restart, and statistics. |
| `src/ter_kernel.cuh` | TER GPU kernel interface and statistics. |
| `src/ter_kernel.cu` | CUDA kernels, device/host buffers, streams, events, and fallback merge. |
| `src/ss_kernel.cuh` | GPU sorting interface for Schroeppel–Shamir. |
| `src/ss_kernel.cu` | CUB radix sort implementation. |
| `src/zero_sum_swap.h` | Subset-based zero-sum swap BFS and solution verification. |
| `src/pairs_tuple.hpp` | Metadata for equal-score pair ranges. |
| `src/profiler.hpp` | RAII-based timing accumulator. |
