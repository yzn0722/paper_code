#!/usr/bin/env python3
"""Plot Fig. 6c: hESC GRN propagation vs edge density.

Observed points are the median Spearman rho over the first eight refinement lags
(same definition as Methods). By default no vertical error bars are shown:
the eight lags belong to one trajectory and are not independent replicates.
Their descriptive spread is retained in source data. Grey shading shows the pooled rewired-null
5th--95th percentile (3 representations x 200 rewirings = 600).

Reads query_to_key_primary results and their rewired draws from the same three
propagation JSON files. Legacy key-to-query results are rejected. Run
upstream/dynamics/run_fig06c_query_to_key.py to generate the matching inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DENSITIES = [1000, 5000, 10000, 20000, 30000]
XLABELS = ["1k", "5k", "10k", "20k", "30k"]
SERIES = (
    ("attn", "attn", "#4EA3F1"),
    ("cos_tok", r"cos$_{tok}$", "#FF9A3D"),
    ("cos_hid", r"cos$_{hid}$", "#AC99D2"),
)

ORIENTATION = "query_to_key_primary"
N_NULL = 200
N_LAGS = 8
DEFAULT_OUTDIR = Path(__file__).resolve().parents[1] / "outputs" / "fig06c" / "query_to_key"
DEFAULT_JSON_DIR = DEFAULT_OUTDIR / "raw"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument(
        "--error",
        choices=("none", "sd", "sem", "iqr"),
        default="none",
        help="Default: no observed error bars. Legacy descriptive lag spread: SD, SEM, or IQR.",
    )
    return p.parse_args()


def load_reports(json_dir: Path) -> dict:
    reports = {}
    for key, _, _ in SERIES:
        path = json_dir / f"weighted_grn_propagation_{key}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["preflight"]["n_null"] != N_NULL:
            raise ValueError(f"{path}: expected {N_NULL} null networks per representation")
        for density in DENSITIES:
            analyses = report["analyses"][str(density)]
            if ORIENTATION not in analyses:
                raise ValueError(
                    f"{path}/{density}: missing {ORIENTATION}; regenerate query-to-key "
                    "results. Legacy key_to_query_primary is not interchangeable."
                )
        reports[key] = report
    shapes = {tuple(r["preflight"]["trajectory_shape"]) for r in reports.values()}
    if len(shapes) != 1:
        raise ValueError("The three representations use different trajectory shapes")
    # Newly generated reports also identify the shared trajectory and gene order.
    provenance = [r.get("input_provenance") for r in reports.values()]
    if any(p is not None for p in provenance):
        if any(p is None for p in provenance):
            raise ValueError("Do not mix reports with and without input provenance")
        for field in ("trajectory_sha256", "expression_sha256", "vocab_sha256"):
            if len({json.dumps(p[field], sort_keys=True) for p in provenance}) != 1:
                raise ValueError(f"The three representations have different {field}")
    return reports


def load_observed_with_lag_errors(json_dir: Path, error: str, reports=None) -> pd.DataFrame:
    reports = load_reports(json_dir) if reports is None else reports
    rows = []
    for key, _, _ in SERIES:
        report = reports[key]
        for density in DENSITIES:
            block = report["analyses"][str(density)][ORIENTATION]["early"]
            lags = block["per_lag"][:N_LAGS]
            if len(lags) != N_LAGS or block["windows"]["transient"]["observed"]["n_lags"] != N_LAGS:
                raise ValueError(f"{key}/{density}: expected 8 transient lags, got {len(lags)}")
            spearman = np.asarray([row["spearman"] for row in lags], dtype=float)
            if not np.isfinite(spearman).all():
                raise ValueError(f"{key}/{density}: non-finite lag Spearman values")
            median = float(np.median(spearman))
            published = float(block["windows"]["transient"]["observed"]["spearman"])
            if abs(median - published) > 1e-10:
                raise ValueError(
                    f"{key}/{density}: median {median} != published window median {published}"
                )
            std = float(spearman.std(ddof=1))
            sem = std / np.sqrt(len(spearman))
            q25 = float(np.percentile(spearman, 25))
            q75 = float(np.percentile(spearman, 75))
            if error == "none":
                yerr_lo = yerr_hi = 0.0
            elif error == "sd":
                yerr_lo = yerr_hi = std
            elif error == "sem":
                yerr_lo = yerr_hi = sem
            else:
                yerr_lo = median - q25
                yerr_hi = q75 - median
            rows.append(
                {
                    "representation": key,
                    "density": density,
                    "transient_spearman": median,
                    "lag_std": std,
                    "lag_sem": sem,
                    "lag_q25": q25,
                    "lag_q75": q75,
                    "yerr_lo": yerr_lo,
                    "yerr_hi": yerr_hi,
                    "n_lags": 8,
                    "error_kind": error,
                    "orientation": ORIENTATION,
                    "group": "early",
                    "representation_null_mean": block["windows"]["transient"]["rewired"]["spearman"]["null_mean"],
                    "representation_empirical_p": block["windows"]["transient"]["rewired"]["spearman"]["empirical_p_greater"],
                }
            )
    return pd.DataFrame(rows)


def load_pooled_null(json_dir: Path, reports=None) -> pd.DataFrame:
    """Pool 3 x 200 draws from the same orientation/window as the observed points."""
    reports = load_reports(json_dir) if reports is None else reports
    rows = []
    for density in DENSITIES:
        draws = []
        for key, _, _ in SERIES:
            block = reports[key]["analyses"][str(density)][ORIENTATION]["early"]
            values = np.asarray(block["windows"]["transient"]["rewired"]["spearman"]["null_values"], float)
            if values.shape != (N_NULL,) or not np.isfinite(values).all():
                raise ValueError(f"{key}/{density}: expected {N_NULL} finite rewired draws")
            draws.append(values)
        pooled = np.concatenate(draws)
        rows.append({
            "top_ranked_edges": density,
            "rewired_null_mean": float(pooled.mean()),
            "rewired_null_q05": float(np.quantile(pooled, 0.05)),
            "rewired_null_q95": float(np.quantile(pooled, 0.95)),
            "rewired_null_n_pooled": len(pooled),
        })
    return pd.DataFrame(rows)


def plot_figure(observed: pd.DataFrame, null: pd.DataFrame, outdir: Path, error: str) -> Path:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 16,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    # Retain the original panel dimensions (180.09 x 118.45 mm).
    fig, ax = plt.subplots(figsize=(7.0903194444, 4.6632361111))
    x = np.arange(len(DENSITIES))

    q05 = null.rewired_null_q05.to_numpy(float)
    q95 = null.rewired_null_q95.to_numpy(float)
    null_mean = null.rewired_null_mean.to_numpy(float)

    ax.fill_between(
        x,
        q05,
        q95,
        color="#B8B8B8",
        alpha=0.35,
        linewidth=0,
        label="Rewired null 5th–95th percentile",
        zorder=1,
    )
    ax.plot(
        x,
        null_mean,
        "--o",
        color="#777777",
        linewidth=2.4,
        markersize=8,
        label="Rewired null mean",
        zorder=2,
    )

    for key, label, color in SERIES:
        sub = observed.loc[observed.representation == key].set_index("density").loc[DENSITIES]
        y = sub.transient_spearman.to_numpy(float)
        style = dict(
            color=color,
            linewidth=2.4,
            markersize=8,
            alpha=0.90,
            label=label,
            zorder=3,
        )
        if error == "none":
            ax.plot(x, y, "-o", **style)
        else:
            yerr = np.vstack([sub.yerr_lo.to_numpy(float), sub.yerr_hi.to_numpy(float)])
            ax.errorbar(x, y, yerr=yerr, fmt="-o", capsize=3.5,
                        capthick=1.2, elinewidth=1.2, **style)

    ax.axhline(0, color="#AAAAAA", linewidth=0.9, zorder=0)
    ax.set_xticks(x, XLABELS)
    ax.set_xlabel(
        "Number of top-ranked GRN edges",
        fontsize=16,
        fontweight="normal",
        labelpad=0,
    )
    ax.set_ylabel(r"Spearman $\rho$", fontsize=16, fontweight="normal", labelpad=0)

    all_lo = np.concatenate(
        [
            observed.transient_spearman.to_numpy(float) - observed.yerr_lo.to_numpy(float),
            q05,
        ]
    )
    all_hi = np.concatenate(
        [
            observed.transient_spearman.to_numpy(float) + observed.yerr_hi.to_numpy(float),
            q95,
        ]
    )
    y_min = min(-0.08, float(np.floor(all_lo.min() / 0.05) * 0.05 - 0.02))
    y_max = max(0.28, float(np.ceil(all_hi.max() / 0.05) * 0.05 + 0.02))
    ax.set_ylim(y_min, y_max)

    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(length=0, labelsize=14, colors="#000000")
    ax.tick_params(axis="x", pad=1)
    ax.grid(False)

    handles, labels = ax.get_legend_handles_labels()
    # fill, null mean, attn, cos_tok, cos_hid
    legend_style = dict(
        frameon=False,
        ncol=3,
        loc="lower center",
        fontsize=16,
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=0.8,
        labelspacing=0.3,
        borderaxespad=0,
    )
    fig.legend(
        [handles[i] for i in [2, 3, 4]],
        [labels[i] for i in [2, 3, 4]],
        bbox_to_anchor=(0.5, 0.055),
        **legend_style,
    )
    fig.legend(
        [handles[i] for i in [1, 0]],
        [labels[i] for i in [1, 0]],
        bbox_to_anchor=(0.5, 0.0),
        **{**legend_style, "ncol": 2},
    )

    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.26, top=0.978)
    outdir.mkdir(parents=True, exist_ok=True)
    suffix = "no_errorbars" if error == "none" else f"err_{error}"
    stem = outdir / f"grn_representation_combined_to30k_with_pooled_null_{suffix}"
    fig.canvas.draw()
    layout = {
        "alignment": "NOT APPLICABLE: one plot panel",
        "observed_error_kind": error,
        "vertical_errorbar_container_count": sum(
            container.__class__.__name__ == "ErrorbarContainer" for container in ax.containers
        ),
        "figure_size_pt": (fig.get_size_inches() * 72).tolist(),
        "plot_area_bounds_fraction": list(ax.get_position().bounds),
    }
    (outdir / "fig06c_layout.json").write_text(json.dumps(layout, indent=2) + "\n", encoding="utf-8")
    fig.savefig(stem.with_suffix(".pdf"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)
    return stem


def main() -> None:
    args = parse_args()
    reports = load_reports(args.json_dir)
    observed = load_observed_with_lag_errors(args.json_dir, args.error, reports)
    null = load_pooled_null(args.json_dir, reports)
    args.outdir.mkdir(parents=True, exist_ok=True)

    source = observed.merge(
        null.rename(
            columns={
                "top_ranked_edges": "density",
                "rewired_null_mean": "pooled_null_mean",
                "rewired_null_q05": "pooled_null_q05",
                "rewired_null_q95": "pooled_null_q95",
                "rewired_null_n_pooled": "pooled_null_n",
            }
        ),
        on="density",
        how="left",
    )
    source_path = args.outdir / "fig06c_density_source_data.csv"
    source.to_csv(source_path, index=False)
    null.to_csv(args.outdir / "fig06c_query_to_key_pooled_null.csv", index=False)
    provenance = {
        "orientation": ORIENTATION,
        "group": "early",
        "transient_lags": N_LAGS,
        "nulls_per_representation": N_NULL,
        "observed_error_kind": args.error,
        "null_display": "pooled mean line and 5th-95th percentile shading; no vertical error bars",
        "observed_error_rationale": "Eight dependent lags of one trajectory, not independent experimental replicates",
        "source_json_sha256": {
            key: hashlib.sha256((args.json_dir / f"weighted_grn_propagation_{key}.json").read_bytes()).hexdigest()
            for key, _, _ in SERIES
        },
        "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.outdir / "fig06c_plot_manifest.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    stem = plot_figure(observed, null, args.outdir, args.error)
    print(f"Wrote {source_path}")
    print(f"Wrote {stem}.pdf / .png / .svg")
    print(
        f"Observed error bars: {args.error} across n_lags=8 "
        f"(point = median Spearman over those lags)."
    )


if __name__ == "__main__":
    main()
