# `adaptssp` CLI

This document describes every command and option defined by the `adaptssp` program. The source of the CLI definition is `src/main.cpp:927-1019`, while argument processing, validation, setup, and solver invocation live in `src/main.cpp:1021-1193`.

## Overview

- `adaptssp` does not use subcommands. All options are passed directly to a single binary.
- The positional argument `INSTANCE` is required.
- The default solver is `ter`.
- Use `--solver ter` or `--solver ss` to select a solver.
- All options can still be parsed for either solver; irrelevant options are not always rejected, but they may be ignored.
- The code reads no application-specific environment variables.

General syntax:

```text
./build/adaptssp INSTANCE [OPTIONS]
```

Shortest example:

```bash
./build/adaptssp work/instance_n24.prb
```

That command uses the TER solver, one attempt, and all other default options.

## Option usage matrix

Legend:

- **Yes**: the option affects that solver's path.
- **Indirect**: the option is accepted and may affect setup, but is not a primary solver control.
- **Ignored**: the option is not read on that solver's path.
- **After solution**: it only runs if the first solver actually produces a solution.

| Argument or option | TER | SS | Notes |
|---|---:|---:|---|
| `INSTANCE` | Yes | Yes | Positional path to the `.prb` file; required. |
| `--solver` | Selector | Selector | Default value `ter`. |
| `-t`, `--threads` | Yes | Yes | Sets `omp_set_num_threads`. |
| `--check_only` | Yes | Yes | Stops the program before the solver call. |
| `--max_pairs` | Ignored | Yes | Limits pair chunks in SS; not a primary TER capacity. |
| `--mem_budget_gb` | Indirect | Indirect | Used to auto-calibrate `--max_pairs`. |
| `--k_radius` | Ignored | Yes | Restricts the cardinalities SS tries. |
| `--runs` | Yes | Ignored | Number of TER attempts. |
| `--autorestart` | Yes | Ignored | Unbounded TER restarts until a solution. |
| `--timeout` | Yes | Ignored | Total TER time limit. |
| `--ter_stats` | Yes | Ignored | Per-attempt TER statistics. |
| `--ter_cpu_frac` | Yes | Ignored | CPU share setting for the TER L1 join. |
| `--ter_b2_extra` | Yes | Ignored | Adds to the TER `b2` parameter. |
| `--ter_heartbeat` | Yes | Ignored | TER progress heartbeat. |
| `--ss_gpu` | Ignored | Yes | CUB quarter sorting for SS, only on the `uint64_t` path. |
| `--extsol` | After solution | After solution | Zero-sum swap exploration. |
| `--extsol_swap_size` | After solution | After solution | Subset size per side, 1 to 6. |
| `--extsol_max_solutions` | After solution | After solution | Limit on the number of `.extsol` solutions. |

Flag-style options are written without a value, for example `--autorestart` or `--extsol`. Options that require a value are written with a space, for example `--threads 8`. Option names in the program use underscores, such as `--check_only`, not `--check-only`.

## General options

### `INSTANCE`

The positional argument holding the input file path:

```bash
./build/adaptssp work/instance_n64.prb
```

The code reads that file as text and takes the instance name from the file name without its extension. For `work/instance_n64.prb`, the instance name becomes `instance_n64`.

Expected file format:

```text
1 n value_1 value_2 ... value_n target
```

The parser takes every digit it finds in the file. Line breaks and extra spaces therefore cause no problems, but the parser is permissive and does not strictly validate signs, negative values, overflow, or extra tokens.

### `-t N`, `--threads N`

Sets the number of OpenMP threads.

```bash
./build/adaptssp work/instance_n24.prb --threads 8
```

- Default: `0`.
- `0` means using the number of logical CPUs detected by the program.
- A positive value is passed directly to `omp_set_num_threads`.
- This option applies to both TER and SS.
- In TER it can affect the OpenMP CPU side and the CPU/GPU split.
- In SS it affects subset generation and the parts that use OpenMP.
- The program performs no special upper-bound validation; use a value that makes sense for the machine.

Example:

```bash
./build/adaptssp work/instance_n64.prb -t 2
```

### `--check_only`

Only reads and prints instance metadata; it does not run TER or SS.

```bash
./build/adaptssp work/instance_n24.prb --check_only
```

What still happens before exiting:

- Automatic `max_pairs_per_chunk` calibration and printing of its value.
- Thread detection and setup.
- Reading and shape validation of the `.prb` file.
- Printing the item count and target.

What does not happen:

- No TER pool is built.
- No SS subsets are generated.
- No solution search.
- No `.sol` or `.extsol` is written.

`--check_only` is not a solution correctness checker. It does not confirm that the target has a subset, and it does not inspect an existing `.sol` file.

This option stays valid together with both `--solver ter` and `--solver ss`, but the solver choice is never processed because the program exits first.

### `--solver ter`

Selects TER:

```bash
./build/adaptssp work/instance_n24.prb --solver ter
```

The default `--solver` value is also `ter`, so the following command is functionally equivalent:

```bash
./build/adaptssp work/instance_n24.prb
```

### `--solver ss`

Selects Schroeppel–Shamir:

```bash
./build/adaptssp work/instance_n64.prb --solver ss
```

SS splits the input into four quarters, computes the cardinality range, and uses two heaps to find matching subset-sum pairs.

Values other than `ter` are not validated as errors by the code. The solver branch only checks:

```cpp
if (solver == "ter")
```

That means any string other than `ter`—including `--solver bogus`—falls through to the SS path. Use only `ter` or `ss` so the behaviour is predictable.

### `--max_pairs N`

Sets the pair chunk limit used when SS extracts pairs with equal scores.

```bash
./build/adaptssp work/instance_n64.prb --solver ss --max_pairs 1000000
```

Sources of this behaviour:

- The value is stored in `max_pairs_per_chunk`.
- `extract_pairs_from_heap` uses it to split groups of pairs into several chunks.
- This value is not a direct limit on the TER L1 kernel output.
- In TER, `target_L1`, `target_L2`, and their capacity limits are computed separately in `ter_solver.cpp`.
- The help text mentions GPU/CPU, but the current implementation mainly uses this control in SS.

Value behaviour:

- If `--max_pairs` is not given, the program computes it automatically.
- If it is given as `0`, the program replaces it with `1000` and prints a warning.
- If a positive value is given, it is used directly without further validation.

### `--mem_budget_gb G`

Sets the memory budget in GiB used for the automatic `max_pairs` calculation.

```bash
./build/adaptssp work/instance_n64.prb --mem_budget_gb 8
```

- Default: `0`.
- `G > 0` is used as an explicit budget.
- `G <= 0` means automatic RAM detection from `/proc/meminfo`.
- If `/proc/meminfo` is unavailable, the program's fallback value is 8 GiB.
- This option does not limit the total process memory.
- This option does not directly limit the TER L1/L2 pools.
- If `--max_pairs` is given explicitly, `--mem_budget_gb` is not used for automatic calibration because that branch is skipped.

The automatic calibration formula used is:

```text
budget_for_pairs = budget_bytes * 0.6
max_pairs = budget_for_pairs / 64
max_pairs minimum = 1000
```

## TER-specific options

The following options are only processed when `--solver ter` is active. If the SS solver is selected, the values are still parsed but do not affect the SS computation.

### `--runs N`

Sets the number of TER attempts.

```bash
./build/adaptssp work/instance_n24.prb --solver ter --runs 20
```

Behaviour:

- Default: `0`.
- Without `--autorestart`, `0` is processed as a single attempt.
- `N > 0` runs at most `N` attempts.
- If a solution is found on the first attempt, TER stops unless statistics mode is active.
- If there is no solution, the program still stops after the attempt limit.
- `N <= 0` without `--ter_stats` is treated as a single attempt.
- In SS this option is ignored.

`--runs` is not a guarantee that every attempt finds a solution. TER uses a random remainder on each attempt.

### `--autorestart`

Enables TER restarts continuously until a solution is found.

```bash
./build/adaptssp work/instance_n24.prb --solver ter --autorestart
```

While the flag is active, the program changes its internal configuration so that:

- `max_restarts` becomes `0`, meaning restarts are not limited.
- `fixed_runs` becomes `0`.
- A `--runs N` given together with this flag is ignored.
- The program can run forever if there is no solution.
- If a solution is found, the program stops on that attempt.

Use a timeout so experiments do not run without a bound:

```bash
./build/adaptssp work/instance_n24.prb --solver ter --autorestart --timeout 60
```

This option does not work with SS.

### `--timeout SECONDS`

Limits the total TER execution time in seconds.

```bash
./build/adaptssp work/instance_n24.prb --solver ter --timeout 30
```

- Default: `0`, meaning no timeout limit.
- The timer starts when `ter_solve` begins executing.
- The check happens after an attempt or after a running pipeline stage.
- The timeout is not a forced stop; kernels, the root join, or large stages can pass the limit before the next check.
- If `--autorestart` is active, the timeout acts as a bound on restarts.
- This option is ignored in SS.

Negative values are not activated as a timeout by the internal condition; use only non-negative second values.

### `--ter_stats`

Enables TER statistics mode.

```bash
./build/adaptssp work/instance_n24.prb --solver ter --runs 20 --ter_stats
```

Conditions validated by the code:

- `--runs N` is required with `N > 0`.
- It must not be used together with `--autorestart`.
- If the conditions are not met, the program exits with code `1` and does not run the solver.

Statistics mode behaviour:

- The program runs attempts according to the `--runs` limit.
- If a solution is found, the attempts do not stop at the first solution.
- The program keeps the first solution, but still counts successful attempts and statistics up to the run limit.
- `--timeout` can still stop earlier than the requested number of runs.
- Statistics cover L2 time, C sorting, L1, root, list sizes, limits exceeded, and GPU/CPU calls in verbose output.

Example with heartbeat:

```bash
./build/adaptssp work/instance_n64.prb --solver ter --runs 20 --ter_stats --ter_heartbeat 5
```

This option is ignored if the SS solver is selected.

### `--ter_cpu_frac F`

Sets the fraction of L1 input rows processed on the CPU during the level 1 join.

```bash
./build/adaptssp work/instance_n64.prb --solver ter --ter_cpu_frac 0.25
```

- Default: `-1`.
- A value of `-1` enables EMA-based adaptive balancing.
- A value `>= 0` enables manual configuration.
- The recommended range is `0.0` to `1.0`.
- `0.0` means all L1 work that can go through the CPU path is directed to the GPU, subject to capacity and join conditions.
- `1.0` means all L1 work that can go through the CPU path is directed to the CPU.
- Adaptive splitting is only relevant when list A is large enough; small lists may still be processed on the CPU.
- There is no range validation at the CLI level. Values outside the range can produce unexpected behaviour.
- If the binary is built without `WITH_GPU`, there is no level 1 CPU/GPU split and all level 1 work runs on the CPU.
- This option is ignored in SS.

This CPU/GPU setting applies to the current process; the program does not store the fractional setting in a configuration file.

### `--ter_b2_extra B`

Adds to the `b2_delta` value in the TER parameters.

```bash
./build/adaptssp work/instance_n64.prb --solver ter --ter_b2_extra 1
```

- Default: `0`.
- Positive values are used.
- A value of `0` preserves the default behaviour.
- Negative values are ignored by the `> 0` condition.
- After a positive value is applied, the program recomputes the derived parameters and prints `b1` and `b2`.
- This option is experimental for tuning; there is no upper-bound validation at the CLI level.
- This option is ignored in SS.

An extra `b2` can reduce L1 checks, but it can also make the L1 list less likely to contain the needed candidates and require more attempts.

### `--ter_heartbeat SECONDS`

Prints a progress heartbeat while TER runs.

```bash
./build/adaptssp work/instance_n64.prb --solver ter --autorestart --ter_heartbeat 10
```

- Default: `0`, heartbeat disabled.
- A positive value enables `[TER-HB]` output.
- The output contains the time since the solver started and the run number.
- The heartbeat is checked at flow points the code has already reached, not inside each kernel.
- This option is ignored in SS.

## SS-specific options

### `--k_radius R`

Sets the cardinality radius around `peak_k` that SS tries.

```bash
./build/adaptssp work/instance_n64.prb --solver ss --k_radius 0
```

Behaviour:

- Default: `-1`.
- `R < 0` tries the entire cardinality window.
- `R = 0` tries only `peak_k`.
- `R > 0` tries cardinalities in the range `peak_k - R` to `peak_k + R`, still bounded by the valid window.
- `peak_k` comes from the priority profile, not a guarantee that that cardinality has a solution.
- This option is ignored in TER.

Examples:

```bash
# All k in the window
./build/adaptssp work/instance_n64.prb --solver ss --k_radius -1

# Only k at the priority peak
./build/adaptssp work/instance_n64.prb --solver ss --k_radius 0

# Around peak_k: peak_k-2 to peak_k+2
./build/adaptssp work/instance_n64.prb --solver ss --k_radius 2
```

Failure at a given radius only means nothing was found in the range that was tried; it is not proof that the instance has no solution.

### `--ss_gpu`

Enables CUB GPU radix sort for part of the SS steps.

```bash
./build/adaptssp work/instance_n96.prb --solver ss --ss_gpu
```

The implementation:

- It only applies to the `uint64_t` specialisation.
- Sorting is done on `quarter2` ascending and `quarter4` descending.
- GPU sorting is only attempted if each list is larger than `65536` elements.
- `quarter1` and `quarter3` still use the CPU path.
- The heap k-way merge still runs on the CPU.
- If CUB/GPU fails, the code falls back to a CPU `std::sort` path.
- If the input cannot be represented as `uint64_t`/`size_t`, the solver uses the `u128` template; on that path `--ss_gpu` is not forwarded and the flag is ignored.
- The option only has an effect if the binary was compiled with CUDA/CUB support (`WITH_GPU`).
- This option is ignored in TER.

Log output like the following can appear when GPU sorting is attempted:

```text
[SS-GPU] sort quarter2=GPU quarter4=CPU(fallback)
```

## Options that apply to both TER and SS

### `--extsol`

Runs zero-sum swap exploration after the first solution is found.

```bash
./build/adaptssp work/instance_n24.prb --solver ter --extsol
```

or:

```bash
./build/adaptssp work/instance_n64.prb --solver ss --extsol
```

Algorithm:

1. The main solver finds one solution and verifies its sum.
2. The initial solution becomes Tier 0.
3. Selected and unselected items are separated.
4. Subsets of the selected elements and of the unselected elements are matched by equal value sum.
5. New candidates are verified against the target.
6. New solutions are processed at the next tier.
7. The process continues until there is no new frontier or one of the internal limits is reached.

This option does not run an additional solver if the main solver finds nothing. With no solution, the program only prints:

```text
[extsol] dilewati: tidak ada solusi awal.
```

### `--extsol_swap_size S`

Sets the maximum subset size on each side of the swap.

```bash
./build/adaptssp work/instance_n24.prb --extsol --extsol_swap_size 2
```

- Default: `4`.
- The valid accepted values are `1` to `6`.
- If `--extsol` is active and the value is outside the range, the program exits with code `1`.
- If `--extsol` is not active, an invalid value is not validated because the option is not used.
- The option applies to TER and SS, but only runs after a solution is found.

Larger values increase the number of candidate subsets and can raise time and memory cost sharply.

### `--extsol_max_solutions N`

Limits the number of solutions stored in `.extsol`.

```bash
./build/adaptssp work/instance_n24.prb --extsol --extsol_max_solutions 100
```

- Default: `1000`.
- The initial solution counts toward the limit.
- `0` means no limit on the number of solutions.
- The option is only used together with `--extsol`.
- It can be used with either TER or SS.
- Other internal limits stay active even when `N=0`.

Internal limits that are not set through the CLI:

- Maximum 100000 tiers.
- Maximum 5,000,000 subsets per side per node before the node is skipped.

### `--extsol` output

The output file is written with the name:

```text
<instance-name>.extsol
```

Each line is a solution bitstring. The initial solution is written as the first line. Writing is relative to the current working folder, not the input file's folder.

Exploration conclusions printed by the application:

- `closure`: all connected solutions within the swap limit have been found.
- `max_solutions tercapai`: the result was capped by `--extsol_max_solutions`.
- `max_tiers tercapai`: the result was capped by the internal tier limit.
- If any node was skipped because its subset count exceeded an internal limit, the result is also marked capped.

This result does not claim that there are no solutions outside the connected component.

## Command combination recipes

### Inspect input only

```bash
./build/adaptssp work/instance_n24.prb --check_only
```

### Single TER attempt

```bash
./build/adaptssp work/instance_n24.prb --solver ter
```

### Several TER attempts

```bash
./build/adaptssp work/instance_n24.prb --solver ter --runs 10
```

### Unbounded TER with timeout and heartbeat

```bash
./build/adaptssp work/instance_n24.prb \
  --solver ter \
  --autorestart \
  --timeout 60 \
  --ter_heartbeat 5
```

### TER statistics

```bash
./build/adaptssp work/instance_n24.prb \
  --solver ter \
  --runs 20 \
  --ter_stats
```

### TER with manual CPU split

```bash
./build/adaptssp work/instance_n64.prb \
  --solver ter \
  --runs 5 \
  --ter_cpu_frac 0.25
```

### Exhaustive SS over the whole k window

```bash
./build/adaptssp work/instance_n64.prb --solver ss --k_radius -1
```

### SS only at the priority peak

```bash
./build/adaptssp work/instance_n64.prb --solver ss --k_radius 0
```

### SS with GPU sorting

```bash
./build/adaptssp work/instance_n96.prb --solver ss --ss_gpu
```

### Solution exploration from TER

```bash
./build/adaptssp work/instance_n24.prb \
  --solver ter \
  --extsol \
  --extsol_swap_size 4 \
  --extsol_max_solutions 1000
```

### Solution exploration from SS

```bash
./build/adaptssp work/instance_n64.prb \
  --solver ss \
  --k_radius 0 \
  --extsol \
  --extsol_swap_size 2
```

### Limiting threads and the memory pair budget

```bash
./build/adaptssp work/instance_n64.prb \
  --solver ss \
  --threads 8 \
  --mem_budget_gb 8
```

To actually change the automatic calibration result, use `--max_pairs` directly:

```bash
./build/adaptssp work/instance_n64.prb \
  --solver ss \
  --max_pairs 1000000
```

## Important interactions and edge cases

### Options are not validated per solver

The parser accepts all options globally. For example, the following command does not fail merely because it uses TER options together with SS:

```bash
./build/adaptssp work/instance_n64.prb --solver ss --runs 10 --ter_heartbeat 5
```

Result: `--runs` and `--ter_heartbeat` are ignored by the SS path. Conversely, `--k_radius` and `--ss_gpu` are ignored if the `ter` solver is selected.

### `--check_only` stops before extsol

`--check_only` is processed after option validation and instance reading, but before the solver call. Because no solution is searched for, `.extsol` output is never produced in this mode.

### extsol validation is only active with the flag

Validation of `--extsol_swap_size` only runs when `--extsol` is active:

```bash
# Not validated as an extsol size because --extsol is not active
./build/adaptssp work/instance_n24.prb --extsol_swap_size 99

# Produces an error because --extsol is active
./build/adaptssp work/instance_n24.prb --extsol --extsol_swap_size 99
```

### `--max_pairs 0`

An explicit `0` does not mean unlimited. The program replaces it with `1000` and prints a warning.

### `--runs` and `--autorestart`

If both are given, `--autorestart` enables unbounded mode and ignores the run count. An example of invalid options:

```bash
./build/adaptssp work/instance_n24.prb \
  --solver ter \
  --autorestart \
  --runs 5 \
  --ter_stats
```

That command stops with an error because `--ter_stats` must not be used together with `--autorestart`.

### `--ter_stats` with timeout

Statistics mode can still be given a timeout. Under that condition the statistics can finish before the `--runs` count is reached, because the timeout is checked at attempt boundaries.

### Solver string typo

Because only the string `ter` is explicitly checked, a typo such as `--solver tr` or `--solver TER` does not enter TER. Such values enter the SS path. Use exactly the lowercase words the code supports.

### SS `u128` and `--ss_gpu`

Input is read as `u128`, then SS selects `uint64_t` only if the target, all elements, and the total still fit in `size_t`. On 64-bit systems, input with large `u128` values can make the program use the `u128` template. `--ss_gpu` does not turn this path into GPU; GPU sorting is only compiled for the `uint64_t` specialisation.

### Output and the current working folder

The output name comes from the input file name. The binary writes output to the current working folder:

```bash
./build/adaptssp work/instance_n24.prb
```

This command writes `instance_n24.sol` in the folder where the command was run, not in `work/`.

To store output in `work/`, run the binary from that folder or change the working folder before running.

## Validation and exit codes

### Code `1`

The program usually exits with code `1` when:

- The positional `INSTANCE` argument is not given.
- An option is unknown or its value format is invalid.
- The `.prb` file cannot be opened.
- The `.prb` file does not have at least four tokens.
- The dimension token is not `1`.
- There are not enough tokens for `n` items and the target.
- `--extsol` is active with `--extsol_swap_size` outside `1..6`.
- `--ter_stats` is used without `--runs N` with `N > 0` on the TER path.
- `--ter_stats` is used together with `--autorestart` on the TER path.

### Code `0`

The program generally still exits with code `0` even when:

- TER finds no solution within the attempt limit.
- SS finds no solution.
- SS finds no solution at a given radius.
- Zero-sum exploration is skipped because there is no initial solution.
- A solution output file cannot be opened on some output paths.

So the process code is not always a "solution found" status. For automated use, check the output and the `.sol`/`.extsol` files explicitly.

## Quick option summary

| Option | Form | Main purpose |
|---|---|---|
| `INSTANCE` | positional | `.prb` input path. |
| `--solver ter` | selector | TER, the default. |
| `--solver ss` | selector | Schroeppel–Shamir. |
| `--threads N` | value | Number of OpenMP threads. |
| `--check_only` | flag | Parse and metadata only. |
| `--max_pairs N` | value | SS pair chunking. |
| `--mem_budget_gb G` | value | Budget for automatic pair calibration. |
| `--k_radius R` | value | SS cardinality radius. |
| `--runs N` | value | Number of TER attempts. |
| `--autorestart` | flag | Unbounded TER restarts. |
| `--timeout S` | value | TER timeout in seconds. |
| `--ter_stats` | flag | Per-attempt TER statistics. |
| `--ter_cpu_frac F` | value | L1 CPU split setting. |
| `--ter_b2_extra B` | value | Extra `b2` bits. |
| `--ter_heartbeat S` | value | TER progress heartbeat. |
| `--ss_gpu` | flag | CUB quarter sorting for SS. |
| `--extsol` | flag | Zero-sum swap exploration. |
| `--extsol_swap_size S` | value | Swap subset size, 1–6. |
| `--extsol_max_solutions N` | value | Extension solution limit. |

## Implementation sources

- All option definitions: `src/main.cpp:927-1019`.
- Argument processing and errors: `src/main.cpp:1021-1037`.
- Pair and thread calibration: `src/main.cpp:1041-1063`.
- `.prb` input and early exit: `src/main.cpp:1065-1083`.
- TER configuration and invocation: `src/main.cpp:1093-1163`.
- SS configuration and invocation: `src/main.cpp:1164-1184`.
- Zero-sum swap extension integration: `src/main.cpp:1186-1193`.
- SS GPU sorting: `src/main.cpp:674-703` and `src/ss_kernel.cu`.
- Extension internal limits: `src/zero_sum_swap.h:30-49` and `src/zero_sum_swap.h:207-328`.
