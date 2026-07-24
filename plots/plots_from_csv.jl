#=
Benchmark plotting script
=========================
Reads the CSVs under `results/<GPU>/` and writes grouped barplots to
`figures/<GPU>/`, using plot_utils.jl.

    julia --project=plots plots/plots_from_csv.jl              # newest GPU dir
    KF_GPU_TAG=A100 julia --project=plots plots/plots_from_csv.jl

These figures are the ones embedded in the KernelForge.jl README — they live
here, next to the CSVs they are drawn from, so re-running a benchmark never
touches the package's git history.
=#
using Pkg
Pkg.activate(@__DIR__)
include("./plot_utils.jl")

using DataFrames
using CSV
using Printf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# CSVs and figures are both in THIS repo, one level up from plots/.
# KF_RESULTS_ROOT still overrides the CSV location (e.g. to plot a scratch
# run without copying it in).
const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))
const RESULTS_ROOT = get(ENV, "KF_RESULTS_ROOT", joinpath(REPO_ROOT, "results"))
isdir(RESULTS_ROOT) || error("No results directory at $RESULTS_ROOT (override with KF_RESULTS_ROOT=…).")
const FIGURES_ROOT = get(ENV, "KF_FIGURES_ROOT", joinpath(REPO_ROOT, "figures"))

# Default to the most-recently-written results directory so a fresh
# run on a new machine plots itself without editing this file. Override
# with `KF_GPU_TAG=…` (e.g. when you have A40 *and* RTX1000 results
# side-by-side and want to (re)plot A40 specifically).
function _detect_gpu_tag()
    haskey(ENV, "KF_GPU_TAG") && return ENV["KF_GPU_TAG"]
    isdir(RESULTS_ROOT) || return "A40"
    dirs = filter(d -> isdir(joinpath(RESULTS_ROOT, d)), readdir(RESULTS_ROOT))
    isempty(dirs) && return "A40"
    # Pick the dir whose newest CSV is most recent.
    function dir_mtime(d)
        path = joinpath(RESULTS_ROOT, d)
        files = filter(f -> endswith(f, ".csv"), readdir(path))
        isempty(files) ? typemin(Float64) : maximum(mtime(joinpath(path, f)) for f in files)
    end
    return last(sort(dirs, by = dir_mtime))
end
const GPU_TAG = _detect_gpu_tag()
@info "plots_from_csv: RESULTS_ROOT=$RESULTS_ROOT  GPU_TAG=$GPU_TAG  (overrides: KF_RESULTS_ROOT=…, KF_GPU_TAG=…)"

const CSV_PATH_MAPREDUCE = joinpath(RESULTS_ROOT, GPU_TAG, "mapreduce.csv")
const CSV_PATH_SCAN = joinpath(RESULTS_ROOT, GPU_TAG, "scan.csv")
const CSV_PATH_SORT = joinpath(RESULTS_ROOT, GPU_TAG, "sort.csv")
const CSV_PATH_SORT_COLUMNS = joinpath(RESULTS_ROOT, GPU_TAG, "sort_columns_min.csv")

const FIG_DIR = joinpath(FIGURES_ROOT, GPU_TAG)
mkpath(FIG_DIR)

# ---------------------------------------------------------------------------
# Figure headings
#
# Every figure carries its own title: the PNGs are reused in the README, the
# docs and the benchmarks repo, where nothing else says which op / GPU /
# direction-is-better they show.
# ---------------------------------------------------------------------------

function _gpu_display_name()
    info = joinpath(RESULTS_ROOT, GPU_TAG, "device_info.json")
    isfile(info) || return GPU_TAG
    name = get(JSON.parsefile(info), "name", GPU_TAG)
    return "$name  ($GPU_TAG)"
end
const GPU_NAME = _gpu_display_name()

"Bold heading: what is measured, on which GPU."
heading(op) = "$op  —  $GPU_NAME"

"Grey sub-heading: the metric, its direction, and who KF is measured against."
function subheading(metric::Symbol, baselines::AbstractString)
    what = metric === :throughput ?
        "kernel throughput (GB/s read) — higher is better" :
        "kernel time (solid) + host overhead (translucent) — lower is better"
    return "$what   ·   Forge = KernelForge   vs   $baselines"
end

"Baseline roster for the current backend, given the vendor library name."
baselines(df, vendor_desc::AbstractString) =
    "$vendor_desc, AcceleratedKernels (AK), $(array_method(df))"

# ---------------------------------------------------------------------------
# Size filters (nothing = use all sizes found in CSV)
# ---------------------------------------------------------------------------

const SIZES_MAPREDUCE = [10^6, 10^7, 10^8]
const SIZES_SCAN = [10^6, 10^7, 10^8]
const SIZES_SORT = [10^6, 10^7, 10^8]
const SIZES_COPY = nothing
const TOTALS_MATVEC = [10^7, 10^8]
const TOTALS_VECMAT = [10^7, 10^8]

function filter_sizes(df_col, override)
    present = sort(unique(df_col))
    isnothing(override) && return present
    # Keep the override's preferred order; drop sizes the CSV doesn't have.
    kept = [s for s in override if s in present]
    # A bench that skipped most of the preferred grid (A100 scan only ran
    # 1e8 / 1e9) would render as a single lonely panel — fall back to every
    # size the CSV does have rather than throw the data away.
    return length(kept) >= 2 ? kept : present
end

# ---------------------------------------------------------------------------
# Label formatter: integer (no decimals) for matvec / vecmat
# ---------------------------------------------------------------------------

fmt_int(x) = @sprintf("%d", round(Int, x))

# Same, but compact past 4 digits — used for the clipped vendor bars
# (a degenerate rocBLAS gemv cell is 6 digits and would overlap its
# neighbours).
fmt_int_compact(x) = abs(x) >= 10^4 ? format_compact(x) : fmt_int(x)

# sizeof for the dtype strings produced by the bench scripts. Used by
# plot_grouped_barplot_multi to convert Gkeys/s → Gbytes/s.
const _DTYPE_BYTES = Dict(
    "UInt8" => 1, "Int8" => 1, "Bool" => 1,
    "UInt16" => 2, "Int16" => 2,
    "UInt32" => 4, "Int32" => 4, "Float32" => 4,
    "UInt64" => 8, "Int64" => 8, "Float64" => 8,
)
sizeof_dtype(T::AbstractString) = get(_DTYPE_BYTES, T, 1)
sizeof_dtype(T) = 1   # fallback (e.g. when T is an Int K-value column)

np_time_scale(total) = total > 10^7 ? 100.0 : 1.0
np_time_unit(total) = total > 10^7 ? "×100 μs" : "μs"

# ---------------------------------------------------------------------------
# Helper: load and normalize a CSV
# ---------------------------------------------------------------------------

function load_df(path)
    df = CSV.read(path, DataFrame)
    if :type in propertynames(df)
        df = rename(df, :type => :T)
    end
    df.method = replace(df.method, "KernelForge" => "Forge")
    return df
end

# ---------------------------------------------------------------------------
# MapReduce
# ---------------------------------------------------------------------------

let df = load_df(CSV_PATH_MAPREDUCE)
    # Drop the vendor rows for UnitFloat8→Float32: CUB / rocPRIM don't do
    # the UInt8→Float32 conversion, they just sum UInt8s in a UInt32
    # accumulator. Showing that next to KF's full conversion path is
    # comparing different work.
    df = filter(r -> !(r.method == vendor_method(df) && r.T == "UnitFloat8→Float32"), df)

    sizes = filter_sizes(df.n, SIZES_MAPREDUCE)

    # Float64 first (widest type → tallest bars set the panel scale and
    # don't clip the smaller-type groups).
    mr_type_order = ["Float64", "Float32", "UInt8", "UnitFloat8→Float32"]

    # Time view (per-call μs/ms, with overhead bands).
    mr_vendor = vendor_method(df) == "CUB" ? "CUB DeviceReduce" : "rocPRIM reduce"
    fig_time = plot_grouped_barplot_multi(df, sizes;
        method_order=reduce_method_order(df),
        highlight_method="Forge",
        type_order=mr_type_order,
        supertitle=heading("Reduction  —  KernelForge.mapreduce"),
        subtitle=subheading(:time, baselines(df, mr_vendor)),
    )
    save(joinpath(FIG_DIR, "mapreduce_$(GPU_TAG)_comparison.png"), fig_time)

    # Throughput view (Gbytes/s on src read; same scale across types).
    fig_throughput = plot_grouped_barplot_multi(df, sizes;
        method_order=reduce_method_order(df),
        highlight_method="Forge",
        type_order=mr_type_order,
        metric=:throughput,
        label_methods=["Forge", vendor_method(df)],
        label_fmt_fn=format_2digits,
        bytes_per_key_fn=sizeof_dtype,
        supertitle=heading("Reduction  —  KernelForge.mapreduce"),
        subtitle=subheading(:throughput, baselines(df, mr_vendor)),
    )
    save(joinpath(FIG_DIR, "mapreduce_$(GPU_TAG)_throughput.png"), fig_throughput)
    @info "MapReduce figures saved (time + throughput)."
end

# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------

let df = load_df(CSV_PATH_SCAN)
    sizes = filter_sizes(df.n, SIZES_SCAN)

    # Float64 first (mirrors mapreduce). QuaternionF64 appended if present.
    scan_type_order = ["Float64", "Float32", "QuaternionF64"]

    scan_vendor = vendor_method(df) == "CUB" ? "CUB DeviceScan" : "rocPRIM inclusive_scan"
    fig_time = plot_grouped_barplot_multi(df, sizes;
        method_order=reduce_method_order(df),
        highlight_method="Forge",
        type_order=scan_type_order,
        supertitle=heading("Prefix scan  —  KernelForge.scan!"),
        subtitle=subheading(:time, baselines(df, scan_vendor)),
    )
    save(joinpath(FIG_DIR, "scan_$(GPU_TAG)_comparison.png"), fig_time)

    fig_throughput = plot_grouped_barplot_multi(df, sizes;
        method_order=reduce_method_order(df),
        highlight_method="Forge",
        type_order=scan_type_order,
        metric=:throughput,
        label_methods=["Forge", vendor_method(df)],
        label_fmt_fn=format_2digits,
        bytes_per_key_fn=sizeof_dtype,
        supertitle=heading("Prefix scan  —  KernelForge.scan!"),
        subtitle=subheading(:throughput, baselines(df, scan_vendor)),
    )
    save(joinpath(FIG_DIR, "scan_$(GPU_TAG)_throughput.png"), fig_throughput)
    @info "Scan figures saved (time + throughput)."
end

# ---------------------------------------------------------------------------
# Helper: load matvec/vecmat CSV (different structure: n, p columns)
# ---------------------------------------------------------------------------

# `dtype` is REQUIRED, not cosmetic: `plot_npbar*` keys a series on
# (x, method) alone and takes the first matching row, so an unfiltered
# frame silently plots whichever dtype the CSV happens to list first —
# and on AMD the vendor label differs per dtype (`LinearAlgebra` for the
# float gemv, `rocBLAS gemm_ex` for 1-byte), so Forge and the baseline
# could end up on DIFFERENT dtypes.
function load_npdf(path; dtype::String="Float32")
    df = CSV.read(path, DataFrame)
    df = rename(df, :type => :T)
    df.method = replace(df.method, "KernelForge" => "Forge")
    df.total_elements = df.n .* df.p
    return filter(r -> r.T == dtype, df)
end

# ---------------------------------------------------------------------------
# MatVec
# ---------------------------------------------------------------------------

let df = load_npdf(joinpath(RESULTS_ROOT, GPU_TAG, "matvec.csv"))
    totals = filter_sizes(df.total_elements, TOTALS_MATVEC)

    figures = Dict(total => plot_npbar(df, total; x_col=:n, xlabel="n (rows)",
        label_fmt_fn=fmt_int,
        time_scale=np_time_scale(total), time_unit=np_time_unit(total))
                   for total in totals)
    for (total, fig) in figures
        #save(joinpath(FIG_DIR, "matvec_np$(total).png"), fig)
    end

    fig_multi = plot_npbar_multi(df, totals; x_col=:n, xlabel="n (rows)",
        label_fmt_fn=fmt_int_compact, clip_factor=6.0,
        title_suffix="  (Float32)",
        supertitle=heading("Matrix–vector product  —  KernelForge.matvec"),
        subtitle=subheading(:time, "$(vendor_blas_label(vendor_blas_method(df))) gemv") *
                 "   ·   hatched bar = clipped, label is the real value",
        time_scale_fn=np_time_scale, time_unit_fn=np_time_unit)
    save(joinpath(FIG_DIR, "matvec_$(GPU_TAG)_comparison.png"), fig_multi)
    @info "MatVec figures saved."
end

# ---------------------------------------------------------------------------
# VecMat
# ---------------------------------------------------------------------------

let df = load_npdf(joinpath(RESULTS_ROOT, GPU_TAG, "vecmat.csv"))
    totals = filter_sizes(df.total_elements, TOTALS_VECMAT)

    figures = Dict(total => plot_npbar(df, total; x_col=:n, xlabel="n (vector length)",
        label_fmt_fn=fmt_int,
        time_scale=np_time_scale(total), time_unit=np_time_unit(total))
                   for total in totals)
    for (total, fig) in figures
        #save(joinpath(FIG_DIR, "vecmat_np$(total).png"), fig)
    end

    fig_multi = plot_npbar_multi(df, totals; x_col=:n, xlabel="n (vector length)",
        label_fmt_fn=fmt_int_compact, clip_factor=6.0,
        title_suffix="  (Float32)",
        supertitle=heading("Vector–matrix product  —  KernelForge.vecmat"),
        subtitle=subheading(:time, "$(vendor_blas_label(vendor_blas_method(df))) gemv") *
                 "   ·   hatched bar = clipped, label is the real value",
        time_scale_fn=np_time_scale, time_unit_fn=np_time_unit)
    save(joinpath(FIG_DIR, "vecmat_$(GPU_TAG)_comparison.png"), fig_multi)
    @info "VecMat figures saved."
end

# ---------------------------------------------------------------------------
# Sort (single-vector) — same shape as scan/mapreduce
# ---------------------------------------------------------------------------

if isfile(CSV_PATH_SORT)
    let df = load_df(CSV_PATH_SORT)
        sizes = filter_sizes(df.n, SIZES_SORT)

        # Widest types first → tallest bars set the panel scale and the
        # narrower types stay readable on the same axis. Mirrors the
        # convention from mapreduce/scan plots.
        sort_type_order = ["UInt64", "Float64", "UInt32", "Float32"]

        # Time view. The array-library baseline (CUDA.jl's bitonic /
        # AMDGPU+Base) is ~50-100× slower → drop it from the time panel
        # entirely (it stays in the throughput panel for completeness).
        # The remaining three (AK, Forge, vendor) span ~10× of dynamic
        # range, which auto-scale handles cleanly: AK fits at the top and
        # the Forge/vendor delta stays readable below.
        sort_time_methods = ["AK", "Forge", vendor_method(df)]
        sort_vendor = vendor_method(df) == "CUB" ? "CUB DeviceRadixSort" : "rocPRIM radix_sort"
        fig_time = plot_grouped_barplot_multi(df, sizes;
            method_order=sort_time_methods,
            highlight_method="Forge",
            type_order=sort_type_order,
            label_methods=sort_time_methods,
            label_fmt_fn=format_1digit,
            supertitle=heading("Radix sort (keys only)  —  KernelForge.sort!"),
            subtitle=subheading(:time, "$sort_vendor, AcceleratedKernels (AK)"),
        )
        save(joinpath(FIG_DIR, "sort_$(GPU_TAG)_comparison.png"), fig_time)

        # Throughput view. Compresses the dynamic range so the bitonic
        # baseline doesn't dominate — kernel throughput in Gbytes/s.
        # Annotate only Forge/vendor so labels don't overcrowd small bars.
        fig_throughput = plot_grouped_barplot_multi(df, sizes;
            method_order=reduce_method_order(df),
            highlight_method="Forge",
            type_order=sort_type_order,
            metric=:throughput,
            label_methods=["Forge", vendor_method(df)],
            label_fmt_fn=format_2digits,
            bytes_per_key_fn=sizeof_dtype,
            supertitle=heading("Radix sort (keys only)  —  KernelForge.sort!"),
            subtitle=subheading(:throughput, baselines(df, sort_vendor)),
        )
        save(joinpath(FIG_DIR, "sort_$(GPU_TAG)_throughput.png"), fig_throughput)
        @info "Sort figures saved (time + throughput)."
    end
else
    @info "Skip sort plot: $CSV_PATH_SORT not found (run sort_perf_comparison.jl first)."
end

# ---------------------------------------------------------------------------
# Sort-columns — per-dtype grouped bar view, Forge vs CUB vs Thrust.
# Two figures (time in µs, throughput in Gbytes/s) matching the layout
# convention used by sort / mapreduce / scan. The `Forge OEM` and
# `Forge Radix` rows in the CSV are intermediate diagnostics — Forge
# internally dispatches to whichever is faster, so the user-facing
# plot only shows the dispatched `Forge` line against external
# baselines.
# ---------------------------------------------------------------------------

const SORT_COLUMNS_EXTERNAL_SPECS = [
    ("Forge",  BENCH_COLORS["Forge"]),
    ("CUB",    BENCH_COLORS["CUB"]),
    ("Thrust", BENCH_COLORS["AK"]),
]

if isfile(CSV_PATH_SORT_COLUMNS)
    let df = load_df(CSV_PATH_SORT_COLUMNS)
        if :T in propertynames(df) && eltype(df.T) <: AbstractString
            rename!(df, :T => :type)
        end

        df_ext = filter(r -> r.method in ("Forge", "CUB", "Thrust"), df)
        if isempty(df_ext)
            @info "Skip sort-columns plots: no CUB / Thrust rows in $CSV_PATH_SORT_COLUMNS."
        else
            fig_t = plot_sort_columns_bars(df_ext;
                metric = :time,
                method_specs = SORT_COLUMNS_EXTERNAL_SPECS,
                supertitle = heading("Segmented sort  —  KernelForge.sort_columns!") *
                             "   (K × M ≈ 4 M keys per cell)",
                # Thrust loop on Float64 at small K (K=64 M=65536 →
                # 65 k serial thrust::sort launches) hits ~6.6 s and
                # would collapse the Forge/CUB bars (~750 µs) to a
                # 1-pixel sliver. Clip Thrust bars at 2× the slowest
                # Forge/CUB result and label them with the actual
                # value (compact-formatted so "6597000" → "6.6M").
                clip_factor = 2.0,
                clip_reference_methods = ["Forge", "CUB"],
                label_fmt_fn = format_compact,
            )
            Label(fig_t[end + 1, 1:2],
                  "CUB = DeviceSegmentedSort::SortKeys     " *
                  "Thrust = packed (sizeof ≤ 4)  /  loop (Float64)",
                  fontsize=11, halign=:center, padding=(0, 0, 4, 6))
            save(joinpath(FIG_DIR, "sort_columns_min_$(GPU_TAG)_comparison.png"), fig_t)

            fig_g = plot_sort_columns_bars(df_ext;
                metric = :throughput,
                method_specs = SORT_COLUMNS_EXTERNAL_SPECS,
                supertitle = heading("Segmented sort throughput  —  KernelForge.sort_columns!"),
            )
            Label(fig_g[end + 1, 1:2],
                  "CUB = DeviceSegmentedSort::SortKeys     " *
                  "Thrust = packed (sizeof ≤ 4)  /  loop (Float64)",
                  fontsize=11, halign=:center, padding=(0, 0, 4, 6))
            save(joinpath(FIG_DIR, "sort_columns_min_$(GPU_TAG)_throughput.png"), fig_g)

            @info "Sort-columns figures saved (time + throughput)."
        end
    end
else
    @info "Skip sort-columns plot: $CSV_PATH_SORT_COLUMNS not found."
end

# ---------------------------------------------------------------------------
# Copy Bandwidth
# ---------------------------------------------------------------------------
const CSV_PATH_COPY = joinpath(RESULTS_ROOT, GPU_TAG, "copy2.csv")

if isfile(CSV_PATH_COPY)
    let df = load_df(CSV_PATH_COPY)
        fig = plot_copy_bandwidth(df; results_dir = joinpath(RESULTS_ROOT, GPU_TAG))
        save(joinpath(FIG_DIR, "copy_$(GPU_TAG)_bandwidth.png"), fig)
        @info "Copy bandwidth figure saved."
    end
else
    @info "Skip copy plot: $CSV_PATH_COPY not found."
end

@info "All done. Figures in $FIG_DIR"