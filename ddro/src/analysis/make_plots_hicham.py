# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "absl-py",
#     "matplotlib",
#     "polars",
#     "pyarrow",
#     "scipy",
#     "seaborn",
# ]
# ///
"""Publication-quality figures for the CIKM 2026 DocID-analysis paper.

Generates a coherent set of figures backing each of the paper's research
questions:

  RQ1 -> effect of (C, L, V) on retrieval        -> fig_metrics_vs_CL
  RQ2 -> does the hybrid PRQ regime pay off       -> fig_winner_per_cell
  RQ3 -> do training-free intrinsic metrics       -> fig_predictor_ranking,
         predict downstream retrieval ranking ?     fig_intrinsic_*

The data layer is polars-native: CSV ingestion goes through ``pl.scan_csv``
+ lazy expressions, joins use ``how="inner", validate="m:1"``, and derived
columns (``C``, ``L``, ``V``, ``M``, ``robustness_gap``) are built as
``pl.col`` expressions in a single ``with_columns`` pass. Per-cell winner
lookups use ``pl.col(metric).arg_max().over(group)`` window expressions
instead of pandas-style ``groupby().idxmax()`` joins.

seaborn does not natively consume polars frames, so each plotting call
materialises a pandas frame via ``.to_pandas()`` immediately before
``sns.*``. scipy/numpy entry points receive numpy arrays via
``.to_numpy()``.

CLI is built on ``absl.app`` + ``absl.flags``: every column name, join
key set, and IR target list is exposed as a flag so the script can be
retargeted at arbitrary CSV schemas without code edits.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from absl import app, flags
from scipy.stats import kendalltau, pearsonr, spearmanr

# ----------------------------------------------------------------------------
# Flags
# ----------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA = _ROOT / "data"
_DEFAULT_OUT = Path(__file__).parent

# Built-in dataset registry: dataset name -> (analysis_csv_name, ir_csv_name).
# Override via --analysis_csv/--ir_csv to read a custom pair.
_BUILTIN_DATASETS: dict[str, tuple[str, str]] = {
    "msmarco": (
        "all_analysis_results_new_full_xps_with_robustness_CLV.csv",
        "all_ir_results_CLV.csv",
    ),
    "nq": (
        "all_analysis_results_new_full_xps_with_robustness_CLV_nq.csv",
        "all_ir_results_CLV_NQ.csv",
    ),
}

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", str(_DEFAULT_DATA),
                    "Directory containing the analysis and IR CSVs.")
flags.DEFINE_string("out_dir", str(_DEFAULT_OUT),
                    "Directory where per-dataset figure subfolders are written.")
flags.DEFINE_multi_string(
    "dataset", None,
    f"Restrict to a built-in dataset (repeatable). "
    f"Choices: {sorted(_BUILTIN_DATASETS)}. Default: all built-in datasets.")
flags.DEFINE_string(
    "analysis_csv", None,
    "Explicit path to the analysis CSV; used with --ir_csv to override "
    "the dataset->filename mapping. Pair them with --dataset to label the "
    "output subfolder.")
flags.DEFINE_string("ir_csv", None, "Explicit path to the IR CSV.")
flags.DEFINE_list(
    "join_keys",
    "encoding,num_codebooks,nb_subspaces,codebook_size",
    "Comma-separated columns used to inner-join the analysis and IR frames.")
flags.DEFINE_string("col_C", "nb_subspaces",
                    "CSV column to expose as paper notation C (parallel axis).")
flags.DEFINE_string("col_L", "num_codebooks",
                    "CSV column to expose as paper notation L (residual axis).")
flags.DEFINE_string("col_V", "codebook_size",
                    "CSV column to expose as paper notation V (codebook size).")
flags.DEFINE_string("col_encoding", "encoding",
                    "CSV column carrying the DocID family identifier.")
flags.DEFINE_list(
    "ir_targets",
    "MRR@10,NDCG@10,NDCG@100,MAP@20,R@10,R@100,Hit@10",
    "IR metric columns iterated over by per-target figures.")


# ----------------------------------------------------------------------------
# Plot styling (constants -- not exposed as flags)
# ----------------------------------------------------------------------------
METHOD_ORDER = ["pq", "rq-kmeans", "rq-module", "pq-rq"]
METHOD_LABELS = {
    "pq": "PQ",
    "rq-kmeans": "R-Kmeans",
    "rq-module": "R-VQ",
    "pq-rq": "PRQ",
}
# Colorblind-friendly palette; PRQ is red across every paper figure.
METHOD_COLORS = {
    "pq":        "#0173b2",  # blue
    "rq-kmeans": "#de8f05",  # orange
    "rq-module": "#029e73",  # green
    "pq-rq":     "#d55e00",  # red
}
METHOD_MARKERS = dict(zip(METHOD_ORDER, ["o", "^", "s", "D"]))
CODEBOOK_SIZES = [128, 256, 512]
CODEBOOK_AREA = {128: 35, 256: 75, 512: 150}

# Figure widths in inches: acmart sigconf column and text widths.
COL_W = 3.34
TEXT_W = 7.00

sns.set_theme(
    context="paper",
    style="whitegrid",
    font="serif",
    font_scale=0.95,
    rc={
        "axes.edgecolor": "0.25",
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "grid.color": "0.88",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "legend.title_fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    },
)

# Paper notation for intrinsic metrics.
METRIC_LABELS = {
    "mean_hierarchical_shared_sim":      r"$\mathcal{S}^{\mathrm{hier}}$",
    "mean_full_shared_sim":              r"$\mathcal{S}^{\mathrm{full}}$",
    "mean_mix_shared_sim":               r"$\mathcal{S}^{\mathrm{mix}}$",
    "correlation_hierarchical":          r"$\mathcal{C}^{\mathrm{hier}}$",
    "correlation_full":                  r"$\mathcal{C}^{\mathrm{full}}$",
    "ranking_preservation_hierarchical": r"$\mathcal{A}^{\mathrm{hier}}$",
    "ranking_preservation_full":         r"$\mathcal{A}^{\mathrm{full}}$",
    "ranking_preservation_mix":          r"$\mathcal{A}^{\mathrm{mix}}$",
    "mean_entropy":                      r"$\mathcal{H}$",
    "mean_gini":                         r"$\mathcal{G}$",
    "uniqueness_ratio":                  r"$\mathcal{U}$",
}


# ----------------------------------------------------------------------------
# Run context: parametrised column names + output directory
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Context:
    """Per-run context: column names, IR targets, output directory.

    Threaded through every figure function so the script never reads
    module-level mutable state. ``current()`` returns the live binding
    that ``save()`` and ``main()`` cooperate on; figure helpers receive
    it explicitly via ``ctx``.
    """
    out_dir: Path
    col_C: str
    col_L: str
    col_V: str
    col_encoding: str
    join_keys: tuple[str, ...]
    ir_targets: tuple[str, ...]


_CURRENT_CTX: Context | None = None


def _ctx() -> Context:
    assert _CURRENT_CTX is not None, "Context not set; call main() first."
    return _CURRENT_CTX


# ----------------------------------------------------------------------------
# Data layer (polars-native)
# ----------------------------------------------------------------------------
def load(analysis_csv: Path, ir_csv: Path, *,
         join_keys: Sequence[str],
         col_C: str, col_L: str, col_V: str,
         col_encoding: str) -> pl.DataFrame:
    """Load + inner-join the analysis and IR CSVs into a single eager frame.

    Both inputs are scanned lazily; ``encoding`` is stripped of the
    leading whitespace that the IR CSV ships with (the IR file's numeric
    columns have ``" 1"``-style padding that polars coerces fine, but
    the string column needs an explicit strip to match the analysis
    side). Paper-notation aliases ``C``, ``L``, ``V`` and the derived
    columns ``M = C*L`` and ``robustness_gap`` are materialised in a
    single ``with_columns`` pass before collection.
    """
    strip_encoding = pl.col(col_encoding).str.strip_chars()
    analysis = pl.scan_csv(analysis_csv).with_columns(strip_encoding)
    ir = pl.scan_csv(ir_csv).with_columns(strip_encoding)

    # The IR CSV ships values with a leading space after every comma
    # (`pq, 1, 24, 0.43...`); polars has no `skipinitialspace` option so
    # every column except `encoding` arrives as Utf8. Strip whitespace
    # and cast everything numeric-looking to Float64 in a single
    # expression pass, then realign the join keys to the analysis-side
    # dtype (typically Int64) so the inner join's schema lines up.
    ir_schema = ir.collect_schema()
    numeric_cast = [
        pl.col(name).str.strip_chars().cast(pl.Float64, strict=False)
        for name, dtype in ir_schema.items()
        if dtype == pl.Utf8 and name != col_encoding
    ]
    if numeric_cast:
        ir = ir.with_columns(numeric_cast)

    analysis_schema = analysis.collect_schema()
    ir_schema = ir.collect_schema()
    key_recasts = [
        pl.col(key).cast(analysis_schema[key], strict=False)
        for key in join_keys
        if ir_schema[key] != analysis_schema[key]
    ]
    if key_recasts:
        ir = ir.with_columns(key_recasts)

    return (
        analysis
        .join(ir, on=list(join_keys), how="inner", validate="m:1")
        .with_columns(
            pl.col(col_C).alias("C"),
            pl.col(col_L).alias("L"),
            pl.col(col_V).alias("V"),
        )
        .with_columns(
            (pl.col("C") * pl.col("L")).alias("M"),
            (pl.col("MRR@10_best") - pl.col("MRR@10_worst"))
                .alias("robustness_gap"),
        )
        .collect()
    )


def load_ir_only(ir_csv: Path, *,
                 col_C: str, col_L: str, col_V: str,
                 col_encoding: str) -> pl.DataFrame:
    """Load IR CSV without joining to analysis data.
    
    Used for plots that only need IR metrics (e.g., C-L heatmaps).
    All rows from the IR CSV are included.
    """
    strip_encoding = pl.col(col_encoding).str.strip_chars()
    ir = pl.scan_csv(ir_csv).with_columns(strip_encoding)
    
    # Strip whitespace and cast numeric columns
    ir_schema = ir.collect_schema()
    numeric_cast = [
        pl.col(name).str.strip_chars().cast(pl.Float64, strict=False)
        for name, dtype in ir_schema.items()
        if dtype == pl.Utf8 and name != col_encoding
    ]
    if numeric_cast:
        ir = ir.with_columns(numeric_cast)
    
    return (
        ir
        .with_columns(
            pl.col(col_C).alias("C"),
            pl.col(col_L).alias("L"),
            pl.col(col_V).alias("V"),
        )
        .with_columns(
            (pl.col("C") * pl.col("L")).alias("M"),
        )
        .collect()
    )


def _method_subset(df: pl.DataFrame, encoding: str) -> pl.DataFrame:
    return df.filter(pl.col("encoding") == encoding)


def _drop_nulls_pd(df: pl.DataFrame, subset: Sequence[str]):
    """Polars ``drop_nulls`` then convert to pandas for seaborn."""
    return df.drop_nulls(subset=list(subset)).to_pandas()


# ----------------------------------------------------------------------------
# Plot primitives
# ----------------------------------------------------------------------------
def annotate_corr(ax, x, y, loc="lower right"):
    r_k, p_k = kendalltau(x, y)
    r_s, p_s = spearmanr(x, y)
    txt = (f"Kendall τ = {r_k:.3f} (p={p_k:.1e})\n"
           f"Spearman ρ = {r_s:.3f} (p={p_s:.1e})")
    ax.text(
        0.98 if "right" in loc else 0.02,
        0.04 if "lower" in loc else 0.96,
        txt,
        transform=ax.transAxes,
        ha="right" if "right" in loc else "left",
        va="bottom" if "lower" in loc else "top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.85),
    )


def base_scatter(ax, df: pl.DataFrame, x: str, y: str):
    sns.scatterplot(
        data=df.to_pandas(), x=x, y=y,
        hue="encoding", hue_order=METHOD_ORDER, palette=METHOD_COLORS,
        style="encoding", style_order=METHOD_ORDER, markers=METHOD_MARKERS,
        size="V", size_order=CODEBOOK_SIZES, sizes=CODEBOOK_AREA,
        edgecolor="black", linewidth=0.4,
        alpha=0.95, ax=ax, legend=False,
    )


def regression_band(ax, df: pl.DataFrame, x: str, y: str,
                    fit_min_y: float | None = 0.4):
    fit = df.filter(pl.col(y) >= fit_min_y) if fit_min_y is not None else df
    xv_arr = fit.get_column(x).to_numpy()
    yv_arr = fit.get_column(y).to_numpy()
    annotate_corr(ax, xv_arr, yv_arr)
    xs = np.linspace(xv_arr.min(), xv_arr.max(), 50)
    m, b = np.polyfit(xv_arr, yv_arr, 1)
    ax.plot(xs, m * xs + b, color="0.3", lw=1.0, ls="--", alpha=0.8, zorder=0)


def _inset_method_legend(ax, loc="upper left"):
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker=METHOD_MARKERS[m], color="w",
               markerfacecolor=METHOD_COLORS[m],
               markeredgecolor="black", markeredgewidth=0.4,
               markersize=7, label=METHOD_LABELS[m])
        for m in METHOD_ORDER
    ]
    leg = ax.legend(
        handles=handles, loc=loc, fontsize=8.5,
        handlelength=1.0, handletextpad=0.4,
        labelspacing=0.3, borderpad=0.4,
        frameon=True, framealpha=0.9, edgecolor="0.7",
    )
    leg.get_frame().set_linewidth(0.4)
    return leg


def save(fig, name: str):
    fig.savefig(_ctx().out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Intrinsic predictors -- shared by per-family and pooled rankings
# ----------------------------------------------------------------------------
_PREDICTOR_KEYS = [
    "uniqueness_ratio",
    # "mean_full_shared_sim", "mean_hierarchical_shared_sim",
    "mean_mix_shared_sim",
    # "ranking_preservation_full", "ranking_preservation_hierarchical",
    "ranking_preservation_mix",
]


def _available_predictors(df: pl.DataFrame) -> list[tuple[str, str]]:
    cols = set(df.columns)
    return [(k, METRIC_LABELS[k]) for k in _PREDICTOR_KEYS if k in cols]


def _pair_arrays(df: pl.DataFrame, col: str, target: str) -> tuple[np.ndarray, np.ndarray]:
    sub = df.drop_nulls(subset=[col, target])
    return sub.get_column(col).to_numpy(), sub.get_column(target).to_numpy()


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_intrinsic_corr(df: pl.DataFrame):
    """Clustermap of Kendall τ -- rows reordered by hierarchical clustering."""
    sim_cols = [c for c in [
        # "mean_full_shared_sim", "mean_hierarchical_shared_sim",
        # "correlation_full", "correlation_hierarchical",
        # "ranking_preservation_full", "ranking_preservation_hierarchical",
        # "mean_entropy", "mean_gini", "uniqueness_ratio",
        "uniqueness_ratio",
        "mean_entropy",
        "mean_mix_shared_sim",
        "ranking_preservation_mix",
    ] if c in df.columns]
    ir_cols = ["MRR@10", "NDCG@10", "NDCG@100", "MAP@20",
               "R@10", "R@100", "Hit@10"]

    def tau(a: np.ndarray, b: np.ndarray) -> float:
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() < 4:
            return 0.0
        t = kendalltau(a[mask], b[mask]).statistic
        return 0.0 if np.isnan(t) else float(t)

    cols_to_pull = sim_cols + ir_cols
    arr = df.select(cols_to_pull).to_numpy()
    matrix = np.array([
        [tau(arr[:, i], arr[:, len(sim_cols) + j])
         for j in range(len(ir_cols))]
        for i in range(len(sim_cols))
    ])

    import pandas as pd  # localised; only for seaborn's clustermap input
    mat = pd.DataFrame(matrix,
                       index=[METRIC_LABELS[s] for s in sim_cols],
                       columns=ir_cols)
    g = sns.clustermap(
        mat, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"fontsize": 6},
        linewidths=0.3, linecolor="white",
        figsize=(COL_W, COL_W * 1.05),
        col_cluster=False, row_cluster=True,
        dendrogram_ratio=(0.15, 0.0),
        cbar_kws={"label": r"Kendall $\tau$"},
        cbar_pos=(0.02, 0.83, 0.03, 0.13),
    )
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=40, ha="right",
             fontsize=6.5)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=7)
    g.fig.subplots_adjust(top=0.95)
    g.fig.savefig(_ctx().out_dir / "fig_intrinsic_corr.pdf",
                  bbox_inches="tight")
    plt.close(g.fig)


def _predictor_ranking_per_family(df: pl.DataFrame, target: str,
                                  kind: str = "pearson"):
    """Per-family R^2 (Pearson) or Kendall τ bar chart."""
    predictors = _available_predictors(df)
    rows: list[dict] = []
    for method in METHOD_ORDER:
        sub_m = _method_subset(df, method)
        if sub_m.height < 4:
            continue
        for col, label in predictors:
            x, y = _pair_arrays(sub_m, col, target)
            if len(x) < 4:
                continue
            v = (pearsonr(x, y).statistic if kind == "pearson"
                 else kendalltau(x, y).statistic)
            if np.isnan(v):
                continue
            rows.append({
                "method": METHOD_LABELS[method],
                "metric": label,
                "value":  v * v if kind == "pearson" else float(v),
                "sign":   float(np.sign(v)),
            })
    if not rows:
        return

    table = pl.DataFrame(rows)
    metric_order = [lbl for _, lbl in predictors]
    method_order = [METHOD_LABELS[m] for m in METHOD_ORDER
                    if METHOD_LABELS[m] in table.get_column("method").unique().to_list()]
    palette = {METHOD_LABELS[m]: METHOD_COLORS[m] for m in METHOD_ORDER}

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 1.25))
    sns.barplot(
        data=table.to_pandas(),
        y="metric", x="value", hue="method",
        order=metric_order, hue_order=method_order,
        palette=palette, edgecolor="black", linewidth=0.3,
        orient="h", ax=ax,
    )
    if kind == "pearson":
        for r in table.iter_rows(named=True):
            if r["sign"] < 0:
                y_idx = metric_order.index(r["metric"])
                ax.text(r["value"] + 0.01, y_idx, "−",
                        ha="left", va="center", fontsize=8, color="0.2")
        ax.set_xlim(0, 1.0)
        ax.set_xlabel(fr"per-family $R^2$ with {target}")
    else:
        ax.axvline(0, color="0.3", lw=0.6)
        ax.set_xlim(-1.0, 1.0)
        ax.set_xlabel(fr"per-family Kendall $\tau$ with {target}")
    ax.set_ylabel("")
    plt.setp(ax.get_yticklabels(), fontsize=8.5)
    leg = ax.legend(loc="lower right", fontsize=7,
                    handlelength=1.0, handletextpad=0.4,
                    columnspacing=0.6, frameon=True,
                    framealpha=0.9, edgecolor="0.7", ncol=1)
    leg.get_frame().set_linewidth(0.4)
    safe = target.replace("@", "at").replace("/", "_")
    save(fig, f"fig_predictor_ranking_{safe}_{kind}")


def _predictor_ranking_pooled(df: pl.DataFrame, target: str):
    """Pooled Kendall τ across all families -- one bar per intrinsic."""
    predictors = _available_predictors(df)
    # Compute per-predictor (tau, n) by repeatedly pairing with the target;
    # done in numpy after a single `.drop_nulls` per pair, since scipy.kendalltau
    # is fastest on contiguous arrays.
    pairs = [(col, label) + _pair_arrays(df, col, target)
             for col, label in predictors]
    n_max = max((len(x) for *_, x, _ in pairs), default=0)
    rows = []
    for col, label, x, y in pairs:
        if len(x) < 0.6 * n_max:
            # _mix variants exist only for PRQ (n=6 vs n=32). Excluding
            # them keeps the pooled tau comparable across predictors.
            continue
        tau = kendalltau(x, y).statistic
        if np.isnan(tau):
            continue
        rows.append({"metric": label, "value": float(tau), "n": len(x)})
    if not rows:
        return

    table = pl.DataFrame(rows).sort("value")
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.6))
    pos, neg = "#029e73", "#d55e00"
    values = table.get_column("value").to_numpy()
    metrics = table.get_column("metric").to_list()
    colours = [pos if v >= 0 else neg for v in values]
    ax.barh(metrics, values, color=colours, edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="0.3", lw=0.6)
    ax.set_xlim(-0.6, 0.6)
    ax.set_xlabel(fr"pooled Kendall $\tau$ with {target}")
    ax.set_ylabel("")
    plt.setp(ax.get_yticklabels(), fontsize=8.5)
    for label, v in zip(metrics, values):
        off = 0.015 if v >= 0 else -0.015
        ax.text(v + off, label, f"{v:+.2f}",
                ha="left" if v >= 0 else "right",
                va="center", fontsize=7.5, color="0.2")
    safe = target.replace("@", "at").replace("/", "_")
    save(fig, f"fig_predictor_ranking_{safe}_pooled")


def fig_predictor_ranking_per_metric(df: pl.DataFrame):
    """One bar chart per IR target, emitted three ways."""
    for target in _ctx().ir_targets:
        if target not in df.columns:
            continue
        _predictor_ranking_per_family(df, target, kind="pearson")
        _predictor_ranking_per_family(df, target, kind="kendall")
        _predictor_ranking_pooled(df, target)


def fig_alignment_hier_vs_full(df: pl.DataFrame):
    x, y = "ranking_preservation_full", "ranking_preservation_hierarchical"
    fig, ax = plt.subplots(figsize=(COL_W, COL_W))
    base_scatter(ax, df, x, y)
    extent = df.select(
        pl.min_horizontal(pl.col(x).min(), pl.col(y).min()).alias("lo"),
        pl.max_horizontal(pl.col(x).max(), pl.col(y).max()).alias("hi"),
    ).row(0)
    lo, hi = extent[0] - 0.02, extent[1] + 0.02
    ax.plot([lo, hi], [lo, hi], color="0.4", lw=0.7, ls=":", zorder=0)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(METRIC_LABELS[x])
    ax.set_ylabel(METRIC_LABELS[y])
    _inset_method_legend(ax, loc="lower right")
    save(fig, "fig_alignment_hier_vs_full")


# --- C vs L heatmaps --------------------------------------------------------
_HEATMAP_METHODS = ["pq", "rq-kmeans", "pq-rq"]


def _grid_pivot(df: pl.DataFrame, *, values: str, agg: str,
                index_vals: Sequence, col_vals: Sequence):
    """Pivot ``values`` on (L, C), aligned to a full ``index_vals`` x
    ``col_vals`` grid (missing cells become NaN).

    Returns a pandas DataFrame indexed by L (rows) and C (cols),
    suitable for ``sns.heatmap``. The full-grid cross-join makes
    missing-cell handling identical between methods that don't visit
    every cell (PQ at L=1, R-Kmeans at C=1).
    """
    import pandas as pd

    # Polars pivot needs the values column to exist for every (L, C) we
    # want to plot. Build the full grid via a cross-join then left-join
    # the actual aggregated values onto it.
    grid = (
        pl.DataFrame({"L": list(index_vals)})
        .join(pl.DataFrame({"C": list(col_vals)}), how="cross")
    )
    agg_expr = {
        "mean": pl.col(values).mean(),
        "max":  pl.col(values).max(),
    }[agg]
    aggregated = (
        df.group_by(["L", "C"])
        .agg(agg_expr.alias(values))
    )
    long = grid.join(aggregated, on=["L", "C"], how="left")
    wide_pl = (
        long.pivot(on="C", index="L", values=values)
        .sort("L")
    )
    value_cols = [c for c in wide_pl.columns if c != "L"]
    wide_pl = wide_pl.with_columns(
        *(pl.col(c).cast(pl.Float64) for c in value_cols)
    )
    pdf = wide_pl.to_pandas().set_index("L")
    pdf.columns = [int(float(c)) for c in pdf.columns]
    pdf = pdf.reindex(columns=list(col_vals)).astype(float)
    pdf.index.name = "L"
    pdf.columns.name = "C"
    return pdf


def fig_metrics_vs_CL_per_method(df: pl.DataFrame):
    """C-vs-L design-space heatmap, one panel per method."""
    methods = _HEATMAP_METHODS
    sub_all = df.filter(pl.col("encoding").is_in(methods))
    all_L = sorted(sub_all.get_column("L").unique().to_list())
    vmin = float(sub_all.get_column("MRR@10").min()) * 100
    vmax = float(sub_all.get_column("MRR@10").max()) * 100

    per_method_C = {
        m: sorted(_method_subset(df, m).get_column("C").unique().to_list())
        for m in methods
    }
    col_counts = [len(per_method_C[m]) for m in methods]
    width_ratios = col_counts + [0.6]

    fig = plt.figure(figsize=(TEXT_W, TEXT_W * 0.45))
    gs = fig.add_gridspec(1, len(methods) + 1, width_ratios=width_ratios,
                          wspace=0.30)
    axes = [fig.add_subplot(gs[0, i]) for i in range(len(methods))]
    cbar_ax = fig.add_subplot(gs[0, -1])

    for idx, (method, ax) in enumerate(zip(methods, axes)):
        sub = _method_subset(df, method)
        cols = per_method_C[method]
        pivot = _grid_pivot(sub, values="MRR@10", agg="mean",
                            index_vals=all_L, col_vals=cols)
        pivot = pivot * 100  # Convert to percentage
        sns.heatmap(
            pivot, ax=ax, cmap="viridis", vmin=vmin, vmax=vmax,
            annot=True, fmt=".1f", annot_kws={"fontsize": 7},
            linewidths=0.4, linecolor="white", square=False,
            cbar=(idx == len(methods) - 1), cbar_ax=cbar_ax,
            cbar_kws={"label": r"mean MRR@10 (%) over $V$"},
            mask=pivot.isna(),
        )
        for i, lval in enumerate(all_L):
            for j, cval in enumerate(cols):
                cell = pivot.loc[lval, cval]
                if cell != cell:  # NaN check without pandas import
                    ax.text(j + 0.5, i + 0.5, "—",
                            ha="center", va="center", color="0.65",
                            fontsize=7)
        ax.set_title(METHOD_LABELS[method], fontsize=9)
        ax.set_xlabel(r"$C$ (parallel)")
        ax.set_ylabel(r"$L$ (residual)" if idx == 0 else "")
        ax.invert_yaxis()
        # Format tick labels as integers
        ax.set_xticklabels([int(float(t.get_text())) for t in ax.get_xticklabels()], fontsize=7.5)
        if idx == 0:
            ax.set_yticklabels([int(float(t.get_text())) for t in ax.get_yticklabels()], fontsize=7.5)
        else:
            ax.set_yticklabels([], fontsize=0)
        if idx != 0:
            ax.set_yticks([])
    save(fig, "fig_metrics_vs_CL_per_method")


def fig_metrics_vs_CL_combined(df: pl.DataFrame):
    """All three families overlaid onto a single (L, C) heatmap.

    Fill encodes max MRR@10 across families per cell; the cell border
    is coloured by the winning family. The winner is computed with a
    ``pl.col("MRR@10").arg_max().over(["L", "C"])`` window expression
    -- one polars pass, no group-then-merge.
    """
    import matplotlib.patches as mpatches

    sub = df.filter(pl.col("encoding").is_in(_HEATMAP_METHODS))
    all_C = sorted(sub.get_column("C").unique().to_list())
    all_L = sorted(sub.get_column("L").unique().to_list())

    pivot = _grid_pivot(sub, values="MRR@10", agg="max",
                        index_vals=all_L, col_vals=all_C)
    pivot = pivot * 100  # Convert to percentage

    # Window expression: winner per (L, C) is the row whose MRR@10 is the
    # group max; pick its `encoding`. Polars resolves the index within the
    # window with `arg_max().over(...)`, then we materialise the chosen
    # encoding string into a (L, C, encoding) frame and pivot to a grid.
    winners = (
        sub
        .filter(
            pl.int_range(pl.len()).over(["L", "C"]) ==
            pl.col("MRR@10").arg_max().over(["L", "C"])
        )
        .select(["L", "C", "encoding"])
        .unique(subset=["L", "C"])
    )
    grid = (
        pl.DataFrame({"L": all_L})
        .join(pl.DataFrame({"C": all_C}), how="cross")
        .join(winners, on=["L", "C"], how="left")
    )
    winner_wide = (
        grid.pivot(on="C", index="L", values="encoding").sort("L")
        .to_pandas().set_index("L")
    )
    winner_wide.columns = [int(float(c)) for c in winner_wide.columns]
    winner_wide = winner_wide.reindex(columns=all_C)

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.9))
    sns.heatmap(
        pivot, ax=ax, cmap="viridis",
        vmin=float(sub.get_column("MRR@10").min()) * 100,
        vmax=float(sub.get_column("MRR@10").max()) * 100,
        annot=True, fmt=".1f", annot_kws={"fontsize": 7},
        linewidths=0.0, square=False,
        cbar_kws={"label": r"MRR@10 (%)", "shrink": 0.85, "pad": 0.02},
        mask=pivot.isna(),
    )
    for i, lval in enumerate(all_L):
        for j, cval in enumerate(all_C):
            cell = pivot.loc[lval, cval]
            if cell != cell:
                ax.text(j + 0.5, i + 0.5, "—",
                        ha="center", va="center", color="0.65", fontsize=7)
                continue
            family = winner_wide.loc[lval, cval]
            if family is None or (isinstance(family, float) and family != family):
                continue
            ax.add_patch(mpatches.Rectangle(
                (j, i), 1, 1, fill=False,
                edgecolor=METHOD_COLORS[family], linewidth=2.0,
            ))
    ax.set_xlabel(r"$C$ (parallel)")
    ax.set_ylabel(r"$L$ (residual)")
    ax.invert_yaxis()
    # Format tick labels as integers
    ax.set_xticklabels([int(float(t.get_text())) for t in ax.get_xticklabels()], fontsize=7.5)
    ax.set_yticklabels([int(float(t.get_text())) for t in ax.get_yticklabels()], fontsize=7.5)

    handles = [mpatches.Patch(facecolor="white", edgecolor=METHOD_COLORS[m],
                              linewidth=2.0, label=METHOD_LABELS[m])
               for m in _HEATMAP_METHODS]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.02),
              ncol=3, fontsize=7, frameon=False, handlelength=1.2,
              handletextpad=0.4, columnspacing=0.8)
    save(fig, "fig_metrics_vs_CL_combined")


# --- winner per (V, M) cell -------------------------------------------------
def fig_winner_per_cell(df: pl.DataFrame):
    """For each (V, M) cell, colour by the winning family and annotate
    with the best MRR@10 + runner-up gap. The winner and gap are derived
    inside a single ``group_by([V, M]).agg(...)`` pipeline so the per-cell
    logic stays in polars.
    """
    Ms = sorted(df.get_column("M").unique().to_list())
    Vs = CODEBOOK_SIZES

    # For every (V, M) cell, collect best/runner-up MRR + the winning family.
    agg = (
        df
        .filter(pl.col("V").is_in(Vs) & pl.col("M").is_in(Ms))
        .group_by(["V", "M"])
        .agg(
            pl.col("MRR@10").max().alias("best"),
            pl.col("encoding").sort_by("MRR@10", descending=True).first()
                .alias("winner"),
            pl.col("MRR@10").sort(descending=True).gather(1).first()
                .alias("runner"),
        )
        .with_columns((pl.col("best") - pl.col("runner")).alias("gap"))
    )

    code = {m: i for i, m in enumerate(METHOD_ORDER)}
    grid = np.full((len(Ms), len(Vs)), np.nan)
    text = np.empty_like(grid, dtype=object)
    text[:] = "—"
    by_cell = {(row["V"], row["M"]): row for row in agg.iter_rows(named=True)}
    for i, M in enumerate(Ms):
        for j, V in enumerate(Vs):
            cell = by_cell.get((V, M))
            if cell is None or cell["winner"] is None:
                continue
            family = cell["winner"]
            grid[i, j] = code[family]
            gap = cell["gap"]
            gap_str = "" if gap is None or (isinstance(gap, float) and gap != gap) \
                else f"\n$+{gap:.3f}$"
            text[i, j] = f"{METHOD_LABELS[family]}\n{cell['best']:.3f}{gap_str}"

    from matplotlib.colors import BoundaryNorm, ListedColormap
    cmap = ListedColormap([METHOD_COLORS[m] for m in METHOD_ORDER])
    norm = BoundaryNorm(np.arange(-0.5, len(METHOD_ORDER) + 0.5, 1), cmap.N)
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.95))
    ax.imshow(grid, cmap=cmap, norm=norm, aspect="auto", origin="lower",
              alpha=0.75)
    for i in range(len(Ms)):
        for j in range(len(Vs)):
            t = text[i, j]
            ax.text(j, i, t, ha="center", va="center",
                    fontsize=6.5, color="black",
                    fontweight="bold" if t != "—" else "normal")
    ax.set_xticks(range(len(Vs))); ax.set_xticklabels([str(v) for v in Vs])
    ax.set_yticks(range(len(Ms))); ax.set_yticklabels([str(m) for m in Ms])
    ax.set_xlabel(r"codebook size $V$")
    ax.set_ylabel(r"DocID length $M = C\!\cdot\!L$")

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=METHOD_COLORS[m], edgecolor="black",
                     linewidth=0.4, alpha=0.75, label=METHOD_LABELS[m])
               for m in METHOD_ORDER]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=6.5,
              handlelength=0.8, handletextpad=0.3, columnspacing=0.6,
              frameon=False)
    ax.grid(False)
    save(fig, "fig_winner_per_cell")


# --- robustness -------------------------------------------------------------
def fig_robustness(df: pl.DataFrame):
    """MRR@10 best-vs-worst spread per family, jittered."""
    sub = df.drop_nulls(subset=["robustness_gap"])
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    rng = np.random.default_rng(0)
    jitter = pl.Series("xpos_jitter",
                       rng.uniform(-0.15, 0.15, sub.height))
    sub = sub.with_columns(
        pl.col("encoding").replace_strict(order).cast(pl.Float64).alias("xpos_base"),
        jitter,
    ).with_columns((pl.col("xpos_base") + pl.col("xpos_jitter")).alias("xpos"))

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.78))
    for method in METHOD_ORDER:
        s2 = _method_subset(sub, method)
        for cb in CODEBOOK_SIZES:
            s3 = s2.filter(pl.col("V") == cb)
            if s3.is_empty():
                continue
            ax.scatter(
                s3.get_column("xpos").to_numpy(),
                s3.get_column("robustness_gap").to_numpy(),
                marker=METHOD_MARKERS[method],
                color=METHOD_COLORS[method],
                s=CODEBOOK_AREA[cb], alpha=0.9,
                edgecolor="black", linewidth=0.4,
            )
    # Per-method median bar
    medians = (
        sub.group_by("encoding")
        .agg(pl.col("robustness_gap").median().alias("med"))
    )
    for row in medians.iter_rows(named=True):
        method = row["encoding"]
        if method not in order:
            continue
        x0 = order[method]
        ax.hlines(row["med"], x0 - 0.25, x0 + 0.25,
                  colors="black", lw=1.2)
    ax.set_xticks(list(order.values()))
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], fontsize=8)
    ax.set_ylabel("MRR@10 best − worst (reruns)")
    save(fig, "fig_robustness")


# --- partial / conditional correlation analyses -----------------------------
def _partial_spearman(df: pl.DataFrame, x: str, y: str,
                      controls: Sequence[str]) -> float:
    """Partial Spearman ρ controlling for `controls`. NaN-safe; the
    expression pipeline rank-transforms numeric columns inside polars
    before falling out to numpy for the OLS residualisation step."""
    numeric_ctrls = [c for c in controls
                     if c in df.columns
                     and df.schema[c] in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                                          pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                                          pl.Float32, pl.Float64)]
    cat_ctrls = [c for c in controls if c not in numeric_ctrls]
    cols = [x, y] + numeric_ctrls + cat_ctrls
    sub = df.select(cols).drop_nulls()
    if sub.height < 4:
        return float("nan")

    rank_exprs = [pl.col(c).rank().alias(f"_r_{c}") for c in [x, y] + numeric_ctrls]
    sub = sub.with_columns(rank_exprs)

    yr = sub.get_column(f"_r_{y}").to_numpy().astype(float)
    xr = sub.get_column(f"_r_{x}").to_numpy().astype(float)
    Z_cols = [sub.get_column(f"_r_{c}").to_numpy().astype(float).reshape(-1, 1)
              for c in numeric_ctrls]
    for c in cat_ctrls:
        dummies = (sub.select(c).to_dummies(columns=[c], drop_first=True)
                   .to_numpy().astype(float))
        Z_cols.append(dummies)
    Z = np.hstack([np.ones((sub.height, 1))] + Z_cols)

    def residuals(v: np.ndarray) -> np.ndarray:
        beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta

    rx, ry = residuals(xr), residuals(yr)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def fig_partial_corr(df: pl.DataFrame, n_boot: int = 2000, seed: int = 0):
    """Bootstrap-CI pointplot of partial Spearman correlation."""
    intrinsic = [
        (k, METRIC_LABELS[k]) for k in [
            # "mean_hierarchical_shared_sim",
            # "correlation_hierarchical",
            # "ranking_preservation_hierarchical",
            # "mean_full_shared_sim",
            "uniqueness_ratio",
            "mean_entropy",
            "mean_mix_shared_sim",
            "ranking_preservation_mix",
        ] if k in df.columns
    ]
    control_sets = [
        ("marginal",       []),
        (r"$|\,V$",        ["V"]),
        (r"$|\,V, M$",     ["V", "M"]),
        (r"$|\,V, M, $enc", ["V", "M", "encoding"]),
    ]
    target = "MRR@10"
    rng = np.random.default_rng(seed)
    rows = []
    n = df.height
    for col, label in intrinsic:
        for cname, ctrls in control_sets:
            point = _partial_spearman(df, col, target, ctrls)
            boots = np.empty(n_boot)
            for b in range(n_boot):
                idx = rng.integers(0, n, size=n)
                try:
                    boots[b] = _partial_spearman(df[idx.tolist()],
                                                 col, target, ctrls)
                except Exception:
                    boots[b] = np.nan
            lo, hi = np.nanpercentile(boots, [2.5, 97.5])
            rows.append({"metric": label, "control": cname,
                         "rho": point, "lo": lo, "hi": hi})
    R = pl.DataFrame(rows)

    metric_order = [lbl for _, lbl in intrinsic]
    highlight = {METRIC_LABELS["mean_hierarchical_shared_sim"]:      "#1f77b4",
                 METRIC_LABELS["ranking_preservation_hierarchical"]: "#2ca02c"}
    palette = {m: highlight.get(m, "0.55") for m in metric_order}
    x_pos = {c: i for i, (c, _) in enumerate(control_sets)}

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.78))
    for m in metric_order:
        sub = R.filter(pl.col("metric") == m).with_columns(
            pl.col("control").replace_strict(x_pos).cast(pl.Int32).alias("x_int")
        ).sort("x_int")
        xs = sub.get_column("x_int").to_numpy()
        ys = sub.get_column("rho").to_numpy()
        lo = sub.get_column("lo").to_numpy()
        hi = sub.get_column("hi").to_numpy()
        col = palette[m]
        is_highlight = m in highlight
        ax.plot(xs, ys, color=col, lw=1.8 if is_highlight else 0.9,
                marker="o", markersize=6 if is_highlight else 4,
                alpha=1.0 if is_highlight else 0.65,
                label=m, zorder=3 if is_highlight else 2)
        ax.fill_between(xs, lo, hi, color=col,
                        alpha=0.18 if is_highlight else 0.08,
                        linewidth=0, zorder=1)

    ax.axhline(0, color="0.4", lw=0.6, ls="--")
    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels([c for c, _ in control_sets], fontsize=7.5)
    ax.set_xlabel(r"Conditioning set ($V$, $M\!=\!C\!\cdot\!L$)")
    ax.set_ylabel(r"Partial Spearman $\rho$ with MRR@10")
    ax.set_ylim(-1.0, 1.0)
    leg = ax.legend(
        loc="lower left", fontsize=7,
        handlelength=1.4, handletextpad=0.4,
        labelspacing=0.25, borderpad=0.3,
        frameon=True, framealpha=0.9, edgecolor="0.7",
        ncol=2, columnspacing=0.8,
    )
    leg.get_frame().set_linewidth(0.4)
    save(fig, "fig_partial_corr")


def fig_within_method_corr(df: pl.DataFrame):
    """Kendall τ inside each family between each intrinsic and MRR@10."""
    intrinsic = [
        (k, METRIC_LABELS[k]) for k in [
            "uniqueness_ratio",
            "mean_entropy",
            "mean_mix_shared_sim",
            "ranking_preservation_mix",
        ] if k in df.columns
    ]
    rows = []
    for col, label in intrinsic:
        for method in METHOD_ORDER:
            sub = _method_subset(df, method).drop_nulls(subset=[col, "MRR@10"])
            if sub.height < 4:
                continue
            t, _ = kendalltau(sub.get_column(col).to_numpy(),
                              sub.get_column("MRR@10").to_numpy())
            rows.append({"metric": label, "method": METHOD_LABELS[method],
                         "tau": float(t)})
    if not rows:
        return
    bar = pl.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.78))
    sns.barplot(
        data=bar.to_pandas(),
        x="metric", y="tau", hue="method",
        hue_order=[METHOD_LABELS[m] for m in METHOD_ORDER],
        palette={METHOD_LABELS[m]: METHOD_COLORS[m] for m in METHOD_ORDER},
        edgecolor="black", linewidth=0.3, ax=ax,
    )
    ax.axhline(0, color="0.3", lw=0.6)
    ax.set_ylabel(r"Within-method Kendall $\tau$")
    ax.set_xlabel("")
    ax.set_ylim(-1.05, 1.05)
    plt.setp(ax.get_xticklabels(), fontsize=8)
    leg = ax.legend(loc="lower left", ncol=2, fontsize=7,
                    handlelength=1.0, handletextpad=0.4,
                    columnspacing=0.8, labelspacing=0.25, borderpad=0.3,
                    frameon=True, framealpha=0.9, edgecolor="0.7")
    leg.get_frame().set_linewidth(0.4)
    save(fig, "fig_within_method_corr")


def _loo_linear(X: np.ndarray, y: np.ndarray):
    """Leave-one-out predictions from linear regression with intercept."""
    n = len(y)
    preds = np.zeros(n)
    X1 = np.hstack([np.ones((n, 1)), X])
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        beta, *_ = np.linalg.lstsq(X1[mask], y[mask], rcond=None)
        preds[i] = X1[i] @ beta
    r2 = 1 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2)
    tau, _ = kendalltau(preds, y)
    return preds, r2, tau


def fig_predicted_vs_actual(df: pl.DataFrame):
    """Leave-one-out predicted vs actual MRR@10 (univariate vs bivariate)."""
    second = "ranking_preservation_hierarchical"
    sub = df.drop_nulls(subset=["mean_hierarchical_shared_sim",
                                second, "MRR@10"])
    y = sub.get_column("MRR@10").to_numpy()
    x_uni = sub.select(["mean_hierarchical_shared_sim"]).to_numpy()
    x_multi = sub.select(["mean_hierarchical_shared_sim", second]).to_numpy()
    preds_uni, r2_uni, tau_uni = _loo_linear(x_uni, y)
    preds_mul, r2_mul, tau_mul = _loo_linear(x_multi, y)

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, TEXT_W * 0.42),
                             sharey=True)
    for ax, preds, r2, tau, title in [
        (axes[0], preds_uni, r2_uni, tau_uni,
         r"univariate: $\mathcal{S}^{\mathrm{hier}}$"),
        (axes[1], preds_mul, r2_mul, tau_mul,
         r"multivariate: $\mathcal{S}^{\mathrm{hier}} + $" + METRIC_LABELS[second]),
    ]:
        plot_df = sub.with_columns(pl.Series("predicted", preds))
        base_scatter(ax, plot_df, "predicted", "MRR@10")
        lo = min(preds.min(), y.min()) - 0.02
        hi = max(preds.max(), y.max()) + 0.02
        ax.plot([lo, hi], [lo, hi], color="0.4", ls=":", lw=0.8, zorder=0)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Predicted MRR@10  (leave-one-out)")
        ax.set_title(title)
        ax.text(
            0.04, 0.96,
            f"LOO $R^2$ = {r2:+.2f}\nKendall τ = {tau:+.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white",
                      ec="0.6", alpha=0.85),
        )
    axes[0].set_ylabel("Actual MRR@10")
    _inset_method_legend(axes[-1], loc="lower right")
    fig.tight_layout()
    save(fig, "fig_predicted_vs_actual")


# --- bootstrap regression band ----------------------------------------------
HEADLINE_INTRINSICS = [
    ("uniqueness_ratio",                  r"$\mathcal{U}$"),
    ("mean_entropy",                      r"$\mathcal{H}$"),
    ("mean_mix_shared_sim",               r"$\mathcal{S}^{\mathrm{mix}}$"),
    ("ranking_preservation_mix",          r"$\mathcal{A}^{\mathrm{mix}}$"),
]


def _bootstrap_regression(x: np.ndarray, y: np.ndarray,
                          n_boot: int = 500, n_points: int = 80,
                          seed: int = 0):
    rng = np.random.default_rng(seed)
    xs = np.linspace(x.min(), x.max(), n_points)
    preds = np.empty((n_boot, n_points))
    n = len(x)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m, c = np.polyfit(x[idx], y[idx], 1)
        preds[b] = m * xs + c
    return (xs, preds.mean(0),
            np.percentile(preds, 2.5, axis=0),
            np.percentile(preds, 97.5, axis=0))


def fig_intrinsic_scatter(df: pl.DataFrame):
    """2x2 scatter: each headline intrinsic vs MRR@10 with bootstrap band."""
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_W * 0.78, TEXT_W * 0.6),
                             sharey=True)
    for ax, (col, label) in zip(axes.flat, HEADLINE_INTRINSICS):
        sub = df.drop_nulls(subset=[col, "MRR@10"])
        sns.scatterplot(
            data=sub.to_pandas(), x=col, y="MRR@10",
            hue="encoding", hue_order=METHOD_ORDER, palette=METHOD_COLORS,
            style="encoding", style_order=METHOD_ORDER, markers=METHOD_MARKERS,
            edgecolor="black", linewidth=0.4, s=45,
            alpha=0.95, ax=ax, legend=False,
        )
        if sub.height >= 5:
            xv = sub.get_column(col).to_numpy()
            yv = sub.get_column("MRR@10").to_numpy()
            xs, mu, lo, hi = _bootstrap_regression(xv, yv)
            ax.plot(xs, mu, color="0.3", lw=1.0, alpha=0.85, zorder=0)
            ax.fill_between(xs, lo, hi, color="0.4", alpha=0.15, zorder=0)
            rho, _ = spearmanr(xv, yv)
            tau, _ = kendalltau(xv, yv)
            ax.text(0.04, 0.96,
                    f"ρ = {rho:+.2f}\nτ = {tau:+.2f}",
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=6.5, color="0.2",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="0.7", lw=0.4, alpha=0.85))
        ax.set_xlabel(label)
        ax.set_ylabel("MRR@10")
    _inset_method_legend(axes[0, 1], loc="lower right")
    fig.tight_layout()
    save(fig, "fig_intrinsic_scatter")


def fig_robustness_vs_quality(df: pl.DataFrame):
    sub = df.drop_nulls(subset=["robustness_gap", "MRR@10"])
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.85))
    base_scatter(ax, sub, "MRR@10", "robustness_gap")
    if sub.height >= 5:
        xv = sub.get_column("MRR@10").to_numpy()
        yv = sub.get_column("robustness_gap").to_numpy()
        xs, mu, lo, hi = _bootstrap_regression(xv, yv)
        ax.plot(xs, mu, color="0.3", lw=1.0, alpha=0.7, zorder=0)
        ax.fill_between(xs, lo, hi, color="0.4", alpha=0.12, zorder=0)
        rho, _ = spearmanr(xv, yv)
        ax.text(0.96, 0.96, f"ρ = {rho:+.2f}",
                transform=ax.transAxes, va="top", ha="right",
                fontsize=7, color="0.2",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="0.7", lw=0.4, alpha=0.85))
    ax.set_xlabel("MRR@10")
    ax.set_ylabel("MRR@10 best − worst (reruns)")
    _inset_method_legend(ax, loc="upper right")
    save(fig, "fig_robustness_vs_quality")


def fig_length_effect(df: pl.DataFrame):
    """MRR@10 vs DocID length M = C*L, with per-family mean traces."""
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.85))
    sns.scatterplot(
        data=df.to_pandas(), x="M", y="MRR@10",
        hue="encoding", hue_order=METHOD_ORDER, palette=METHOD_COLORS,
        style="encoding", style_order=METHOD_ORDER, markers=METHOD_MARKERS,
        size="V", size_order=CODEBOOK_SIZES, sizes=CODEBOOK_AREA,
        edgecolor="black", linewidth=0.4, alpha=0.85,
        ax=ax, legend=False,
    )
    # Per-method mean per M -- one polars expression per family.
    for m in METHOD_ORDER:
        traces = (
            _method_subset(df, m)
            .group_by("M")
            .agg(pl.col("MRR@10").mean().alias("mean_mrr"))
            .sort("M")
        )
        if traces.height >= 2:
            ax.plot(traces.get_column("M").to_numpy(),
                    traces.get_column("mean_mrr").to_numpy(),
                    color=METHOD_COLORS[m], lw=1.0, alpha=0.6, zorder=0)
    ax.set_xscale("log", base=2)
    Ms = sorted(df.get_column("M").unique().to_list())
    ax.set_xticks(Ms); ax.set_xticklabels([str(m) for m in Ms])
    ax.set_xlabel(r"DocID length $M = C \cdot L$")
    ax.set_ylabel("MRR@10")
    _inset_method_legend(ax, loc="lower right")
    save(fig, "fig_length_effect")


def fig_recall_at_k(df: pl.DataFrame):
    """Recall growth curves per family."""
    ks = [1, 10, 100, 1000]
    cols = [f"R@{k}" for k in ks]
    if not all(c in df.columns for c in cols):
        return
    summary = (
        df.group_by("encoding")
        .agg(
            *[pl.col(c).mean().alias(f"mean_{c}") for c in cols],
            *[pl.col(c).min().alias(f"min_{c}")  for c in cols],
            *[pl.col(c).max().alias(f"max_{c}")  for c in cols],
        )
    )
    by_encoding = {row["encoding"]: row for row in summary.iter_rows(named=True)}

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.85))
    for m in METHOD_ORDER:
        row = by_encoding.get(m)
        if row is None:
            continue
        means = [row[f"mean_{c}"] for c in cols]
        lo = [row[f"min_{c}"] for c in cols]
        hi = [row[f"max_{c}"] for c in cols]
        ax.plot(ks, means, color=METHOD_COLORS[m],
                marker=METHOD_MARKERS[m], markersize=6,
                markeredgecolor="black", markeredgewidth=0.4,
                lw=1.2, label=METHOD_LABELS[m])
        ax.fill_between(ks, lo, hi, color=METHOD_COLORS[m],
                        alpha=0.10, zorder=0)
    ax.set_xscale("log", base=10)
    ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"Recall@$k$  (mean over configurations)")
    leg = ax.legend(loc="lower right", fontsize=7,
                    handlelength=1.2, handletextpad=0.4,
                    frameon=True, framealpha=0.9, edgecolor="0.7")
    leg.get_frame().set_linewidth(0.4)
    save(fig, "fig_recall_at_k")


def fig_intrinsic_per_family(df: pl.DataFrame):
    """Strip + box of each headline intrinsic by family."""
    fig, axes = plt.subplots(1, len(HEADLINE_INTRINSICS),
                             figsize=(TEXT_W, TEXT_W * 0.28),
                             sharex=False)
    for ax, (col, label) in zip(axes, HEADLINE_INTRINSICS):
        sub = df.drop_nulls(subset=[col]).with_columns(
            pl.col("encoding").replace_strict(METHOD_LABELS).alias("enc_label")
        )
        order = [METHOD_LABELS[m] for m in METHOD_ORDER]
        palette = {METHOD_LABELS[m]: METHOD_COLORS[m] for m in METHOD_ORDER}
        pdf = sub.to_pandas()
        sns.boxplot(data=pdf, x="enc_label", y=col, order=order,
                    palette=palette, ax=ax,
                    width=0.55, fliersize=0,
                    boxprops=dict(alpha=0.45, edgecolor="black", linewidth=0.6),
                    medianprops=dict(color="black", linewidth=1.0),
                    whiskerprops=dict(color="0.4", linewidth=0.5),
                    capprops=dict(color="0.4", linewidth=0.5))
        sns.stripplot(data=pdf, x="enc_label", y=col, order=order,
                      palette=palette, ax=ax, size=2.5,
                      jitter=0.15, edgecolor="black", linewidth=0.3)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel(""); ax.set_ylabel("")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=6.5)
        plt.setp(ax.get_yticklabels(), fontsize=6.5)
    save(fig, "fig_intrinsic_per_family")


def fig_intrinsic_pairgrid(df: pl.DataFrame):
    """Pairgrid of the headline intrinsics, points coloured by family."""
    cols = [c for c, _ in HEADLINE_INTRINSICS if c in df.columns]
    labels = {c: lbl for c, lbl in HEADLINE_INTRINSICS}
    sub = (
        df.drop_nulls(subset=cols + ["encoding"])
        .with_columns(
            pl.col("encoding").replace_strict(METHOD_LABELS).alias("enc_label")
        )
    )
    palette = {METHOD_LABELS[m]: METHOD_COLORS[m] for m in METHOD_ORDER}
    g = sns.PairGrid(sub.to_pandas(), vars=cols, hue="enc_label",
                     hue_order=[METHOD_LABELS[m] for m in METHOD_ORDER],
                     palette=palette,
                     height=COL_W / 2.2, aspect=1.0)
    g.map_diag(sns.kdeplot, fill=True, alpha=0.45, lw=0.7)
    g.map_offdiag(sns.scatterplot, s=25, edgecolor="black", linewidth=0.3,
                  alpha=0.85)
    for ax in g.axes.flatten():
        if ax is None:
            continue
        ax.tick_params(labelsize=5.5)
    for i, c in enumerate(cols):
        g.axes[-1, i].set_xlabel(labels[c], fontsize=7)
        g.axes[i, 0].set_ylabel(labels[c], fontsize=7)
    g.add_legend(title="", fontsize=6,
                 label_order=[METHOD_LABELS[m] for m in METHOD_ORDER],
                 bbox_to_anchor=(1.02, 0.5), loc="center left")
    g.fig.savefig(_ctx().out_dir / "fig_intrinsic_pairgrid.pdf",
                  bbox_inches="tight")
    plt.close(g.fig)


def fig_R_metric(df: pl.DataFrame):
    if not {"r_hierarchical_norm", "r_full_norm"}.issubset(df.columns):
        return
    sub = df.drop_nulls(subset=["r_hierarchical_norm", "r_full_norm"])
    fig, ax = plt.subplots(figsize=(COL_W, COL_W))
    base_scatter(ax, sub, "r_full_norm", "r_hierarchical_norm")
    extent = sub.select(
        pl.min_horizontal(pl.col("r_full_norm").min(),
                          pl.col("r_hierarchical_norm").min()).alias("lo"),
        pl.max_horizontal(pl.col("r_full_norm").max(),
                          pl.col("r_hierarchical_norm").max()).alias("hi"),
    ).row(0)
    lo, hi = extent[0] - 0.02, extent[1] + 0.02
    ax.plot([lo, hi], [lo, hi], color="0.4", lw=0.7, ls=":", zorder=0)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$\mathcal{R}^{\mathrm{full}}$ (normalised)")
    ax.set_ylabel(r"$\mathcal{R}^{\mathrm{hier}}$ (normalised)")
    _inset_method_legend(ax, loc="lower right")
    save(fig, "fig_R_metric")


# Plots that only need IR data (no analysis metrics required)
IR_ONLY_PLOTS = [
    fig_metrics_vs_CL_per_method,
    fig_metrics_vs_CL_combined,
]

# Plots that need both IR and analysis data
PLOTS = [
    fig_predictor_ranking_per_metric,
    fig_intrinsic_scatter,
    fig_robustness_vs_quality,
    fig_length_effect,
    fig_recall_at_k,
    fig_intrinsic_per_family,
    fig_intrinsic_pairgrid,
    fig_R_metric,
]


# --- cross-dataset figure ---------------------------------------------------
def fig_cross_dataset(loaded: dict[str, pl.DataFrame], base_ctx: Context):
    """Per-configuration MRR@10 on MSMARCO vs NQ."""
    a = loaded["msmarco"]
    b = loaded["nq"]
    keys = list(base_ctx.join_keys)
    merged = a.join(b, on=keys, how="inner", suffix="_nq")
    # After polars suffixing the left side keeps original names and the right
    # side gets the suffix; rename to match the figure's column expectations.
    merged = merged.rename({"MRR@10": "MRR@10_ms"})

    global _CURRENT_CTX
    out_dir = base_ctx.out_dir / "cross"
    out_dir.mkdir(parents=True, exist_ok=True)
    _CURRENT_CTX = Context(
        out_dir=out_dir, col_C=base_ctx.col_C, col_L=base_ctx.col_L,
        col_V=base_ctx.col_V, col_encoding=base_ctx.col_encoding,
        join_keys=base_ctx.join_keys, ir_targets=base_ctx.ir_targets,
    )

    fig, ax = plt.subplots(figsize=(COL_W, COL_W))
    base_scatter(ax, merged, "MRR@10_ms", "MRR@10_nq")
    extent = merged.select(
        pl.min_horizontal(pl.col("MRR@10_ms").min(),
                          pl.col("MRR@10_nq").min()).alias("lo"),
        pl.max_horizontal(pl.col("MRR@10_ms").max(),
                          pl.col("MRR@10_nq").max()).alias("hi"),
    ).row(0)
    lo, hi = extent[0] - 0.02, extent[1] + 0.02
    ax.plot([lo, hi], [lo, hi], color="0.4", ls=":", lw=0.8, zorder=0)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ms = merged.get_column("MRR@10_ms").to_numpy()
    nq = merged.get_column("MRR@10_nq").to_numpy()
    tau, _ = kendalltau(ms, nq)
    rho, _ = spearmanr(ms, nq)
    ax.text(
        0.96, 0.04,
        f"τ = {tau:.2f}\nρ = {rho:.2f}\nn = {merged.height}",
        transform=ax.transAxes, va="bottom", ha="right", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7",
                  alpha=0.9, lw=0.4),
    )
    ax.set_xlabel("MRR@10 (MSMARCO)")
    ax.set_ylabel("MRR@10 (NQ)")
    _inset_method_legend(ax, loc="upper left")
    save(fig, "fig_cross_dataset")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def _resolve_dataset_inputs(name: str, data_dir: Path,
                            analysis_override: Path | None,
                            ir_override: Path | None) -> tuple[Path, Path]:
    if analysis_override is not None and ir_override is not None:
        return analysis_override, ir_override
    if name not in _BUILTIN_DATASETS:
        raise app.UsageError(
            f"Dataset {name!r} is not built-in; pass --analysis_csv and "
            f"--ir_csv to load a custom pair.")
    a, b = _BUILTIN_DATASETS[name]
    return data_dir / a, data_dir / b


def main(argv):
    del argv
    data_dir = Path(FLAGS.data_dir)
    out_dir = Path(FLAGS.out_dir)

    analysis_csv = Path(FLAGS.analysis_csv) if FLAGS.analysis_csv else None
    ir_csv = Path(FLAGS.ir_csv) if FLAGS.ir_csv else None
    if (analysis_csv is None) != (ir_csv is None):
        raise app.UsageError(
            "--analysis_csv and --ir_csv must be given together.")

    datasets = FLAGS.dataset or list(_BUILTIN_DATASETS)
    join_keys = tuple(FLAGS.join_keys)
    ir_targets = tuple(FLAGS.ir_targets)

    base_ctx_kwargs = dict(
        col_C=FLAGS.col_C, col_L=FLAGS.col_L, col_V=FLAGS.col_V,
        col_encoding=FLAGS.col_encoding,
        join_keys=join_keys, ir_targets=ir_targets,
    )

    global _CURRENT_CTX
    loaded: dict[str, pl.DataFrame] = {}
    for dataset in datasets:
        analysis_path, ir_path = _resolve_dataset_inputs(
            dataset, data_dir, analysis_csv, ir_csv,
        )
        df = load(
            analysis_path, ir_path,
            join_keys=join_keys,
            col_C=FLAGS.col_C, col_L=FLAGS.col_L, col_V=FLAGS.col_V,
            col_encoding=FLAGS.col_encoding,
        )
        loaded[dataset] = df
        
        # Load IR-only data for plots that don't need analysis metrics
        df_ir = load_ir_only(
            ir_path,
            col_C=FLAGS.col_C, col_L=FLAGS.col_L, col_V=FLAGS.col_V,
            col_encoding=FLAGS.col_encoding,
        )

        per_dataset_out = out_dir / dataset
        per_dataset_out.mkdir(parents=True, exist_ok=True)
        _CURRENT_CTX = Context(out_dir=per_dataset_out, **base_ctx_kwargs)

        print(f"\n=== {dataset}: merged rows={df.height}, IR rows={df_ir.height} ===")
        counts = (df.group_by("encoding").len().sort("encoding"))
        for row in counts.iter_rows(named=True):
            print(f"  {row['encoding']:<12s} {row['len']}")
        
        # Run IR-only plots first
        for fn in IR_ONLY_PLOTS:
            fn(df_ir)
            print(f"  {fn.__name__} ✓ (IR-only)")
        
        # Run plots that need both IR and analysis data
        for fn in PLOTS:
            fn(df)
            print(f"  {fn.__name__} ✓")
        print(f"wrote outputs to {per_dataset_out}")

    if set(loaded) >= {"msmarco", "nq"}:
        base_ctx = Context(out_dir=out_dir, **base_ctx_kwargs)
        fig_cross_dataset(loaded, base_ctx)


if __name__ == "__main__":
    app.run(main)
