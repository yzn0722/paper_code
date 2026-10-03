

#!/usr/bin/env python3
"""Plot cumulative direction accuracy across iterations for three mHSC-L starts."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import Bbox


ROOT = Path("/mnt/10T/yzn/scGRN-Bench")
RESULT = ROOT / "outputs/tf_cross_dataset_iter8/mHSC-L"
OUT = RESULT / "figures"
STYLES = {
    "early": ("Early start", "#2E6F9E"),
    "middle": ("Intermediate start", "#D98B2B"),
    "late": ("Late start", "#3E9B76"),
}


def balanced_direction_accuracy(delta: np.ndarray, truth: np.ndarray) -> float:
    valid = np.abs(truth) > 1e-8
    predicted = np.sign(delta[valid])
    observed = np.sign(truth[valid])
    recalls = []
    for direction in (-1, 1):
        class_mask = observed == direction
        if np.any(class_mask):
            recalls.append(np.mean(predicted[class_mask] == direction))
    return float(np.mean(recalls))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = json.load(open(RESULT / "dynamic_grn_validation.json"))
    paths = report["preflight"]["paths"]
    expr = pd.read_csv(paths["expression"], index_col=0)
    pt = pd.read_csv(paths["pseudotime"])
    pt = pt.rename(columns={pt.columns[0]: "cell", pt.columns[1]: "pt"}).set_index("cell")
    common = expr.columns.intersection(pt.index)
    expr = expr[common]
    pseudotime = pt.loc[common, "pt"].to_numpy(float)
    x = np.clip(expr.T.to_numpy(np.float32), 0, None)
    vmax = max(float(np.percentile(x, 99.5)), 1e-6)
    x = np.clip(x / vmax * 50.0, 0, 50.0)
    vocab = json.load(open(Path(paths["model"]) / "vocab.json"))
    mapped = np.asarray([str(g).upper() in vocab for g in expr.index])
    if int(mapped.sum()) != int(report["n_mapped_genes"]):
        raise RuntimeError("Mapped-gene count mismatch")

    q20, q80 = np.quantile(pseudotime, [0.2, 0.8])
    early_center = x[pseudotime <= q20][:, mapped].mean(0)
    late_center = x[pseudotime >= q80][:, mapped].mean(0)
    true_delta = late_center - early_center
    top_n = max(int(len(true_delta) * 0.30), 1)
    top = np.argsort(np.abs(true_delta))[::-1][:top_n]
    truth = true_delta[top]

    rows = []
    curves = {}
    for key, (label, color) in STYLES.items():
        states = np.load(RESULT / f"{key}_mean_trajectory.npy")[:, mapped][:, top]
        cumulative = []
        for t in range(1, len(states)):
            cumulative_acc = balanced_direction_accuracy(states[t] - states[0], truth)
            step_acc = balanced_direction_accuracy(states[t] - states[t - 1], truth)
            cumulative.append(cumulative_acc)
            rows.append({"start": label, "iteration": t,
                         "cumulative_balanced_direction_accuracy": cumulative_acc,
                         "stepwise_balanced_direction_accuracy": step_acc,
                         "n_top_genes": top_n})
        curves[key] = np.asarray(cumulative)

    # Match convergence_models_hESC.pdf exactly: page size, axes footprint,
    # typography, strokes, markers, legend density, and outer whitespace.
    page_w = 344.809 / 72.0
    page_h = 326.252 / 72.0
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "axes.linewidth": 1.2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "pdf.fonttype": 3,
        "ps.fonttype": 3,
    })
    fig = plt.figure(figsize=(page_w, page_h))
    ax = fig.add_axes([0.1537, 0.1243, 0.8192, 0.8515])
    for key, (label, color) in STYLES.items():
        iterations = np.arange(1, len(curves[key]) + 1)
        ax.plot(iterations, curves[key] * 100.0, color=color, lw=2.4,
                marker="o", ms=8, markeredgewidth=0, alpha=0.85,
                label=label)
    ax.set(xlabel="Iteration", ylabel="Balanced Accuracy (%)")
    ax.set_xlim(1, 8)
    ax.set_xticks([1, 3, 5, 7])
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))

    # ===== 新增：在 50% 处画一条横向虚线 =====
    ax.axhline(
        y=50.0,
        color="gray",      # 颜色，可改成 "#888888" / "black" 等
        linestyle="--",    # 虚线
        linewidth=1.2,     # 线宽
        alpha=0.8,         # 透明度
        zorder=0,          # 画在曲线下方，避免遮挡
    )
    # =======================================

    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right", ncol=1,
              handlelength=1.6, handletextpad=0.5, labelspacing=0.3,
              borderaxespad=0.3, columnspacing=0.8)
    stem = OUT / "mHSC-L_start_state_direction_accuracy"
    fixed_page = Bbox.from_bounds(0, 0, page_w, page_h)
    fig.savefig(stem.with_suffix(".pdf"), dpi=600,
                bbox_inches=fixed_page, pad_inches=0)
    fig.savefig(stem.with_suffix(".png"), dpi=600,
                bbox_inches=fixed_page, pad_inches=0)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(stem.with_suffix(".csv"), index=False)
    print(stem.with_suffix(".pdf"))
    print("top genes", top_n)
    for key, (label, _) in STYLES.items():
        print(label, np.round(curves[key], 4).tolist())


if __name__ == "__main__":
    main()


