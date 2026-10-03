#!/usr/bin/env python3
"""Joint mHSC-L UMAP fit on every observed and generated cell."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import FancyArrowPatch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path("/mnt/10T/yzn/benchmark_GRN")
EXPR_CSV = ROOT / "input_process/CHIP/mHSC-L_chip_matched-ExpressionData.csv"
PT_CSV = ROOT / "PseudoTime/mHSC-L/PseudoTime.csv"
OUTDIR = ROOT / "pre_scgpt/0204code/Pseudotime_trajectory_plots_umap"
PRED_NPY = OUTDIR / "mHSC-L_scgpt_early_trueLate_predLate_joint_umap_predicted_cells.npy"
OUTPUT = OUTDIR / "mHSC-L_scgpt_allcells_joint_umap_curves_adjusted_spacing"

PT_QUANTILE = 0.2
SEED = 42
COLORS = {
    "Observed Early": "#E69F00",
    "Observed Late": "#4EA3F1",
    "scGPT Late-like": "#FF9A3D",
}


def bin_expr_to_0_50(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0)
    x = np.clip(x, 0, None)
    vmax = max(float(np.percentile(x, 99.5)), 1e-6)
    return np.clip(x / vmax * 50.0, 0, 50).astype(np.float32)


def main() -> None:
    expr = pd.read_csv(EXPR_CSV, index_col=0)
    pt_df = pd.read_csv(PT_CSV)
    pt_df = pt_df.rename(columns={pt_df.columns[0]: "cell", pt_df.columns[1]: "pt"}).set_index("cell")
    common = expr.columns.intersection(pt_df.index)
    expr = expr[common]
    pt = pd.to_numeric(pt_df.loc[common, "pt"], errors="coerce").to_numpy(float)
    valid = np.isfinite(pt)
    expr, pt = expr.loc[:, valid], pt[valid]
    x_all = bin_expr_to_0_50(expr.T.to_numpy(dtype=np.float32))

    lo, hi = np.quantile(pt, [PT_QUANTILE, 1.0 - PT_QUANTILE])
    early_mask, late_mask = pt <= lo, pt >= hi
    middle_mask = (~early_mask) & (~late_mask)
    x_early, x_late, x_middle = x_all[early_mask], x_all[late_mask], x_all[middle_mask]
    x_pred = np.load(PRED_NPY).astype(np.float32)

    # Fit every preprocessing and embedding step jointly, including the real
    # intermediate-pseudotime cells. No out-of-sample UMAP transform is used.
    combined = np.vstack([x_early, x_middle, x_late, x_pred]).astype(np.float32)
    scaler = StandardScaler().fit(combined)
    scaled = scaler.transform(combined)
    n_pcs = min(50, scaled.shape[0] - 1, scaled.shape[1])
    pca = PCA(n_components=n_pcs, random_state=SEED).fit(scaled)
    pcs = pca.transform(scaled)
    embedding = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2,
        metric="euclidean", random_state=SEED,
    ).fit_transform(pcs)

    n_early, n_middle, n_late = len(x_early), len(x_middle), len(x_late)
    i1 = n_early
    i2 = i1 + n_middle
    i3 = i2 + n_late
    middle_embedding = embedding[i1:i2]
    clouds = {
        "Observed Early": embedding[:i1],
        "Observed Late": embedding[i2:i3],
        "scGPT Late-like": embedding[i3:],
    }
    centers = {label: cloud.mean(axis=0) for label, cloud in clouds.items()}

    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.scatter(
        middle_embedding[:, 0], middle_embedding[:, 1],
        s=19, alpha=0.38, color="#9B9B9B", edgecolors="none",
        label="Observed Intermediate cells",
        rasterized=True, zorder=0,
    )

    start = centers["Observed Early"]
    for target_label, curvature in (("Observed Late", -0.16), ("scGPT Late-like", 0.16)):
        target_center = centers[target_label]
        toward_start = start - target_center
        toward_start = toward_start / np.linalg.norm(toward_start)
        projections = (clouds[target_label] - target_center) @ toward_start
        boundary_radius = np.quantile(projections, 0.92)
        target_edge = target_center + boundary_radius * toward_start
        ax.add_patch(FancyArrowPatch(
            start, target_edge,
            connectionstyle=f"arc3,rad={curvature}", arrowstyle="-|>",
            mutation_scale=15, linewidth=2.3, color=COLORS[target_label],
            alpha=0.88, shrinkA=12, shrinkB=2, zorder=1,
        ))

    for label, color in COLORS.items():
        cloud = clouds[label]
        ax.scatter(
            cloud[:, 0], cloud[:, 1], s=34, alpha=0.78, color=color,
            label={
                "Observed Early": "Observed Early cells",
                "Observed Late": "Observed Late cells",
                "scGPT Late-like": "Predicted Late cells",
            }[label], edgecolors="none",
            rasterized=True, zorder=2,
        )
        centroid = centers[label]
        ax.scatter(
            centroid[0], centroid[1], s=135, marker="X", color=color,
            edgecolors="white", linewidths=1.0, zorder=5,
        )

    ax.set_xlabel("UMAP 1", fontsize=16)
    ax.set_ylabel("UMAP 2", fontsize=16)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis="both", which="both", labelsize=16, length=0)
    ax.margins(x=0.02, y=0.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.legend(
        frameon=True, facecolor="white", framealpha=0.92, edgecolor="none",
        fontsize=14, loc="upper left",
        handletextpad=0.3, labelspacing=0.3, borderaxespad=0.12,
    )
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.085, top=0.94)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUTPUT.with_suffix(".pdf"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"early={len(x_early)} middle={len(x_middle)} late={len(x_late)} predicted={len(x_pred)}")
    print(OUTPUT)


if __name__ == "__main__":
    main()
