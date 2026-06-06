# KernelForge-benchmarks

Benchmark result snapshots for [KernelForge.jl](https://github.com/epilliat/KernelForge.jl).

This repo holds CSV outputs from the benchmark scripts under
`KernelForge.jl/perfs/julia/benchmarks/` and `KernelForge.jl/perfs/cuda_cpp/`.
Splitting them out of the main repo keeps the package's git history small —
the CSVs change every time a benchmark is re-run, and that churn doesn't
belong in source history.

## Layout

```
results/
  <GPU>/
    mapreduce.csv
    scan.csv
    sort.csv
    sort_columns.csv
    sort_columns_min.csv
    matvec.csv
    vecmat.csv
    random.csv
    randperm.csv
    device_info.json
    ...
```

GPU directories present: `RTX1000` (Turing TU117, current), `A40` (Ampere
GA102, historical), `MI300X` (CDNA3, historical).

## Schemas

- 1D kernels (`mapreduce`, `scan`, `sort`, `random`, `randperm`):
  `n, type, method, mean_kernel_μs, std_kernel_μs, mean_total_μs, std_total_μs`
- 2D kernels (`matvec`, `vecmat`): adds `n, p` (rows × cols), drops `n`.
- Batched (`sort_columns*`): `K, M, type, method, ...` (K columns of length M).

`device_info.json` is the runtime device snapshot from
`KernelForge.jl/perfs/julia/bench_utils.jl`.

## Adding a result

Run a benchmark from a checkout of `KernelForge.jl` — output goes to
`perfs/julia/results/<GPU>/<bench>.csv` by default. Move that file here
into `results/<GPU>/`, commit, push.

## Plotting

The plotter is back in the main repo at
`KernelForge.jl/perfs/julia/plots_from_csv.jl`. By default it reads
`../KernelForge-benchmarks/results/<GPU>/` (sibling-repo layout). Override
the root via `KF_RESULTS_ROOT=/path/to/results` if your clone lives
elsewhere.
