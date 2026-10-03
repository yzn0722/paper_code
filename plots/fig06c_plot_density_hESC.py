#!/usr/bin/env python3
"""Plot Fig. 6c: hESC GRN propagation vs edge density, with per-point error bars.

Observed points are the median Spearman rho over the first eight refinement lags
(same definition as Methods). Error bars show the sample s.d. (ddof=1) of those
eight lag-wise Spearman values. Grey shading remains the pooled rewired-null
5th--95th percentile (3 representations x 200 rewirings = 600).

Reads archived propagation JSON under scGRN-Bench/FBplot/fig6/raw_rewiring/ and
writes figures + source CSV under paper-code/outputs/fig06c/.
"""

from __future__ import annotations

import argparse
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

DEFAULT_JSON_DIR = Path(
    "/mnt/10T/yzn/scGRN-Bench/FBplot/fig6/raw_rewiring"
)
DEFAULT_PLOT_DATA = Path(
    "/mnt/10T/yzn/scGRN-Bench/FBplot/fig6/plot_data.csv"
)
DEFAULT_OUTDIR = Path(__file__).resolve().parents[1] / "outputs" / "fig06c"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR)
    p.add_argument("--plot-data", type=Path, default=DEFAULT_PLOT_DATA)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument(
        "--error",
        choices=("sd", "sem", "iqr"),
        default="sd",
        help="Error bars on observed points: lag sample s.d. (default), SEM, or IQR.",
    )
    return p.parse_args()


def load_observed_with_lag_errors(json_dir: Path, error: str) -> pd.DataFrame:
    rows = []
    for key, _, _ in SERIES:
        path = json_dir / f"weighted_grn_propagation_{key}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        for density in DENSITIES:
            block = report["analyses"][str(density)]["key_to_query_primary"]["early"]
            lags = block["per_lag"][:8]
            if len(lags) != 8:
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
            if error == "sd":
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
                }
            )
    return pd.DataFrame(rows)


def load_pooled_null(plot_data: Path) -> pd.DataFrame:
    frame = pd.read_csv(plot_data)
    need = {
        "top_ranked_edges",
        "rewired_null_mean",
        "rewired_null_q05",
        "rewired_null_q95",
        "rewired_null_n_pooled",
    }
    if not need.issubset(frame.columns):
        raise ValueError(f"plot_data.csv missing columns: {sorted(need - set(frame.columns))}")
    frame = frame.set_index("top_ranked_edges").loc[DENSITIES].reset_index()
    if not (frame.rewired_null_n_pooled == 600).all():
        raise ValueError("Expected pooled null n=600 at each density")
    return frame


def plot_figure(observed: pd.DataFrame, null: pd.DataFrame, outdir: Path, error: str) -> Path:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 3,
        }
    )
    fig, ax = plt.subplots(figsize=(510.503 / 72.0, 335.753 / 72.0))
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
    ax.errorbar(
        x,
        null_mean,
        yerr=np.vstack([null_mean - q05, q95 - null_mean]),
        fmt="--o",
        color="#777777",
        linewidth=2.4,
        markersize=8,
        capsize=3.5,
        capthick=1.2,
        elinewidth=1.2,
        label="Rewired null mean",
        zorder=2,
    )

    for key, label, color in SERIES:
        sub = observed.loc[observed.representation == key].set_index("density").loc[DENSITIES]
        y = sub.transient_spearman.to_numpy(float)
        yerr = np.vstack([sub.yerr_lo.to_numpy(float), sub.yerr_hi.to_numpy(float)])
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="-o",
            color=color,
            linewidth=2.4,
            markersize=8,
            alpha=0.90,
            capsize=3.5,
            capthick=1.2,
            elinewidth=1.2,
            label=label,
            zorder=3,
        )

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
    stem = outdir / f"grn_representation_combined_to30k_with_pooled_null_err_{error}"
    fig.savefig(stem.with_suffix(".pdf"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)
    return stem


def main() -> None:
    args = parse_args()
    observed = load_observed_with_lag_errors(args.json_dir, args.error)
    null = load_pooled_null(args.plot_data)
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
    source_path = args.outdir / "fig06c_density_with_lag_errorbars_source_data.csv"
    source.to_csv(source_path, index=False)

    stem = plot_figure(observed, null, args.outdir, args.error)
    print(f"Wrote {source_path}")
    print(f"Wrote {stem}.pdf / .png / .svg")
    print(
        f"Observed error bars: {args.error} across n_lags=8 "
        f"(point = median Spearman over those lags)."
    )


if __name__ == "__main__":
    main()
