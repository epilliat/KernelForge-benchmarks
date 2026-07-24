# KernelForge-benchmarks

Benchmark result snapshots for [KernelForge.jl](https://github.com/epilliat/KernelForge.jl).

This repo holds the CSV outputs from the benchmark scripts under
`KernelForge.jl/perfs/julia/benchmarks/` and `KernelForge.jl/perfs/cuda_cpp/`,
**plus the figures drawn from them and the plotting code that draws them**.
Splitting all three out of the main repo keeps the package's git history
small — CSVs and PNGs change every time a benchmark is re-run, and that churn
doesn't belong in source history. The KernelForge.jl README embeds the
figures below straight from this repo, so refreshing one is a commit here and
nothing in the package.

## Layout

```
results/            raw benchmark output, one dir per GPU
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
figures/            PNGs drawn from results/, same GPU dirs
  <GPU>/
    <bench>_<GPU>_comparison.png     time view   (embedded in the KF README)
    <bench>_<GPU>_throughput.png     GB/s view
plots/              the plotting code + its own Julia environment
  plots_from_csv.jl   results/<GPU>/*.csv  ->  figures/<GPU>/*.png
  plot_utils.jl       figure primitives (grouped bars, np bars, headings)
  number_format.jl    number formatting helpers
  sort_svg_options.jl alternative sort-figure styles (SVG + PNG)
  Project.toml        CairoMakie / DataFrames / CSV / JSON
```

GPU directories present: `A100` (Ampere SXM4 80 GB), `MI300A` (CDNA3 APU),
`RTX1000` (Ada laptop, day-to-day dev), `A40` (Ampere GA102, historical),
`MI300X` (CDNA3, historical).

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

No GPU needed — the plotter only reads CSVs.

```
julia --project=plots -e "using Pkg; Pkg.instantiate()"   # first time only
KF_GPU_TAG=A100 julia --project=plots plots/plots_from_csv.jl
```

Without `KF_GPU_TAG` it picks the `results/` directory with the most recently
written CSV. `KF_RESULTS_ROOT` / `KF_FIGURES_ROOT` override the input and
output roots.

Each figure carries its own title, GPU name (read from `device_info.json`)
and metric, because the PNGs are viewed far from this repo. The vendor
baseline is resolved from the CSV, not assumed: `CUB` / `cuBLAS` / `CUDA` on
NVIDIA, `rocPRIM` / `LinearAlgebra` (rocBLAS `gemv`) / `Base` on AMD.

### Embedding a figure

The KernelForge.jl README links them by raw URL, pinned to `main`:

```
https://raw.githubusercontent.com/epilliat/KernelForge-benchmarks/main/figures/<GPU>/<file>.png
```

Re-running the plotter and pushing updates the README images in place — no
change needed in the package repo. (GitHub caches images through camo, so a
refreshed figure can take a few minutes to appear.)
