#!/usr/bin/env python3
"""Compute and plot cross-dataset GRN/dynamics coupling for scGPT.

The calculation matches the hESC density plot: absolute expression change at
iteration t is propagated through an incoming-normalized GRN and compared with
the absolute change at t+1 using Spearman correlation.  The reported transient
score is the median across the first eight available lags.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATASETS = ["hESC", "hHep", "mDC", "mHSC-E", "mHSC-GM", "mHSC-L"]
DENSITIES = [1000, 5000, 10000, 20000, 30000]
NETWORKS = {
    "attn": ("output_att500", "#4EA3F1"),
    "COS_tok": ("output_emb500", "#FF9A3D"),
    "COS_hid": ("output_embhidden500", "#AC99D2"),
}
DISPLAY = {"attn": "attn", "COS_tok": r"cos$_{tok}$", "COS_hid": r"cos$_{hid}$"}

# Legacy key->query figure-source values; invalid under query->key orientation.
# Prefer --no-use-hesc-reference (now the default) to recompute hESC.
HESC_REFERENCE = {
    "attn": [-0.0439733538, 0.0533248487, 0.0891385699, 0.0749321230, 0.0422636298],
    "COS_tok": [0.1388477806, 0.1414005656, 0.1447418305, 0.1907019138, 0.2180762328],
    "COS_hid": [0.2004276884, 0.2411240527, 0.2099518950, 0.1779605604, 0.1564664486],
}
HESC_NULL_REFERENCE = {
    "null_mean": -0.0018345895,
    "null_q05": -0.0331181454,
    "null_q95": 0.0301016365,
    "n_null": 600,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark-root", default="/mnt/10T/yzn/benchmark_GRN")
    p.add_argument("--trajectory-root", default="/mnt/10T/yzn/scGRN-Bench/outputs/dynamic_grn_cross_dataset")
    p.add_argument("--outdir", default="/mnt/10T/yzn/scGRN-Bench/outputs/grn_density_cross_dataset")
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--transient-lags", type=int, default=8)
    p.add_argument("--bar-density", type=int, default=10000)
    p.add_argument("--n-null", type=int, default=200,
                   help="Degree-preserving rewired networks per representation at bar density.")
    p.add_argument("--seed", type=int, default=20260924)
    p.add_argument("--plot-only", action="store_true",
                   help="Render the chart from existing observed and rewired-null CSV files.")
    hesc = p.add_mutually_exclusive_group()
    hesc.add_argument("--use-hesc-reference", dest="use_hesc_reference", action="store_true")
    hesc.add_argument("--no-use-hesc-reference", dest="use_hesc_reference", action="store_false")
    p.set_defaults(use_hesc_reference=False)
    return p.parse_args()


def find_case_insensitive(folder: Path, dataset: str) -> Path:
    candidates = [p for p in folder.glob("*.tsv") if p.stem.lower() == f"scgpt_{dataset}".lower()]
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one scGPT TSV for {dataset} in {folder}, found {candidates}")
    return candidates[0]


def expression_path(root: Path, dataset: str) -> Path:
    return root / "input_process" / "CHIP" / f"{dataset}_chip_matched-ExpressionData.csv"


def score_dataset(
    dataset: str,
    benchmark: Path,
    trajectory_root: Path,
    transient_lags: int,
    null_density: int,
    n_null: int,
    seed: int,
):
    from run_weighted_grn_propagation import (
        configuration_rewire,
        load_grn,
        normalize_incoming,
        score_trajectory,
        summarize_window,
    )

    expr = expression_path(benchmark, dataset)
    trajectory = trajectory_root / dataset / "early_mean_trajectory.npy"
    vocab_path = benchmark / "pre_scgpt" / "scGPT" / "scgpt_human" / "vocab.json"
    missing = [p for p in (expr, trajectory, vocab_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{dataset}: missing {missing}")

    genes = [str(g).strip().upper() for g in pd.read_csv(expr, index_col=0, usecols=[0]).index]
    with vocab_path.open(encoding="utf-8") as f:
        allowed = set(genes) & {str(g).strip() for g in json.load(f)}
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    states = np.asarray(np.load(trajectory), dtype=float)
    if states.ndim != 2 or states.shape[1] != len(genes):
        raise ValueError(f"{dataset}: trajectory shape {states.shape} does not match {len(genes)} genes")
    if states.shape[0] < transient_lags + 2:
        raise ValueError(
            f"{dataset}: need at least {transient_lags + 2} states for {transient_lags} lag pairs; "
            f"found {states.shape[0]}"
        )

    rows: list[dict] = []
    pooled_null: list[float] = []
    evl = benchmark / "evl_omipath"
    for network_index, (network, (folder_name, _)) in enumerate(NETWORKS.items()):
        grn_path = find_case_insensitive(evl / folder_name / "scgpt", dataset)
        all_edges = load_grn(grn_path, allowed, gene_to_idx, max(DENSITIES))
        if len(all_edges) < max(DENSITIES):
            raise ValueError(f"{dataset}/{network}: only {len(all_edges)} valid positive edges")
        for density in DENSITIES:
            edge = all_edges.head(density)
            # Attention TSV stores Gene1=query, Gene2=key; use query -> key.
            src = edge.Gene1.map(gene_to_idx).to_numpy(np.int32)
            dst = edge.Gene2.map(gene_to_idx).to_numpy(np.int32)
            raw = edge.EdgeWeight.to_numpy(float)
            weight = normalize_incoming(dst, raw, states.shape[1])
            per_lag = score_trajectory(states, src, dst, weight, 0.10)
            score = summarize_window(per_lag, 0, transient_lags)["spearman"]
            rows.append(
                {
                    "dataset": dataset,
                    "network": network,
                    "density": density,
                    "transient_spearman": score,
                    "n_lags": transient_lags,
                    "trajectory": str(trajectory),
                    "grn": str(grn_path),
                }
            )
            if density == null_density:
                rng = np.random.default_rng(seed + network_index * 1009)
                for _ in range(n_null):
                    rewired_src, rewired_dst, rewired_raw, _ = configuration_rewire(
                        src, dst, raw, rng
                    )
                    rewired_weight = normalize_incoming(
                        rewired_dst, rewired_raw, states.shape[1]
                    )
                    rewired_lags = score_trajectory(
                        states, rewired_src, rewired_dst, rewired_weight, 0.10
                    )
                    value = summarize_window(
                        rewired_lags, 0, transient_lags
                    )["spearman"]
                    if value is not None and np.isfinite(value):
                        pooled_null.append(float(value))
    if not pooled_null:
        raise RuntimeError(f"{dataset}: no finite rewired-null values")
    null = np.asarray(pooled_null, dtype=float)
    null_summary = {
        "dataset": dataset,
        "density": null_density,
        "null_mean": float(null.mean()),
        "null_q05": float(np.quantile(null, 0.05)),
        "null_q95": float(np.quantile(null, 0.95)),
        "n_null": int(len(null)),
    }
    return rows, null_summary


def hesc_reference_rows() -> list[dict]:
    return [
        {
            "dataset": "hESC",
            "network": network,
            "density": density,
            "transient_spearman": value,
            "n_lags": 8,
            "trajectory": "figure-source reference",
            "grn": "figure-source reference",
        }
        for network, values in HESC_REFERENCE.items()
        for density, value in zip(DENSITIES, values)
    ]


def plot_bar(frame: pd.DataFrame, null_frame: pd.DataFrame, outdir: Path, density: int) -> None:
    data = frame[frame.density == density].copy()
    wanted = [d for d in DATASETS if d != "mDC" and d in set(data.dataset)]
    x = np.arange(len(wanted), dtype=float)
    width = 0.155
    spacing = 0.18
    offsets = [-1.5 * spacing, -0.5 * spacing, 0.5 * spacing]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 16,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 3,
    })
    fig, ax = plt.subplots(figsize=(7.090319444, 4.663236111))
    for offset, (network, (_, color)) in zip(offsets, NETWORKS.items()):
        values = []
        for dataset in wanted:
            hit = data[(data.dataset == dataset) & (data.network == network)]
            values.append(float(hit.transient_spearman.iloc[0]) if len(hit) == 1 else np.nan)
        ax.bar(x + offset, values, width=width, color=color, alpha=0.85, label=DISPLAY[network])
    null_means, null_q05, null_q95 = [], [], []
    for dataset in wanted:
        hit = null_frame[(null_frame.dataset == dataset) & (null_frame.density == density)]
        if len(hit) != 1:
            raise ValueError(f"{dataset}: expected one null summary at density {density}")
        null_means.append(float(hit.null_mean.iloc[0]))
        null_q05.append(float(hit.null_q05.iloc[0]))
        null_q95.append(float(hit.null_q95.iloc[0]))
    null_means = np.asarray(null_means)
    null_q05 = np.asarray(null_q05)
    null_q95 = np.asarray(null_q95)
    ax.bar(
        x + 1.5 * spacing,
        null_means,
        width=width,
        color="#A8A8A8",
        alpha=0.85,
        yerr=np.vstack([null_means - null_q05, null_q95 - null_means]),
        error_kw={"ecolor": "#333333", "elinewidth": 1.0, "capsize": 3, "capthick": 1.0},
        label="Rewired null",
    )
    ax.axhline(0, color="#AAAAAA", linewidth=0.9)
    ax.set_xticks(x, wanted)
    ax.set_ylabel(r"Spearman $\rho$", fontsize=16, fontweight="normal", labelpad=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(length=0, labelsize=14, colors="#000000")
    ax.tick_params(axis="x", pad=1)
    ax.grid(False)
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        fontsize=16,
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=0.8,
        labelspacing=0.3,
        borderaxespad=0,
    )
    fig.subplots_adjust(left=0.17, right=0.985, top=0.978, bottom=0.26)
    stem = outdir / f"grn_transient_spearman_cross_dataset_bar_top{density // 1000}k"
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    benchmark = Path(args.benchmark_root)
    trajectory_root = Path(args.trajectory_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        frame = pd.read_csv(outdir / "grn_transient_spearman_cross_dataset_all_densities.csv")
        null_frame = pd.read_csv(outdir / "grn_transient_spearman_cross_dataset_rewired_null.csv")
        plot_bar(frame, null_frame, outdir, args.bar_density)
        return
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    rows: list[dict] = []
    null_rows: list[dict] = []
    if args.n_null < 1:
        raise ValueError("--n-null must be positive")
    for dataset_index, dataset in enumerate(datasets):
        if dataset == "hESC" and args.use_hesc_reference:
            rows.extend(hesc_reference_rows())
            null_rows.append({"dataset": "hESC", "density": args.bar_density,
                              **HESC_NULL_REFERENCE})
        else:
            print(f"Scoring {dataset}", flush=True)
            dataset_rows, null_summary = score_dataset(
                dataset,
                benchmark,
                trajectory_root,
                args.transient_lags,
                args.bar_density,
                args.n_null,
                args.seed + dataset_index * 100003,
            )
            rows.extend(dataset_rows)
            null_rows.append(null_summary)
    frame = pd.DataFrame(rows)
    null_frame = pd.DataFrame(null_rows)
    frame.to_csv(outdir / "grn_transient_spearman_cross_dataset_all_densities.csv", index=False)
    null_frame.to_csv(outdir / "grn_transient_spearman_cross_dataset_rewired_null.csv", index=False)
    plot_bar(frame, null_frame, outdir, args.bar_density)
    shown = frame[frame.density == args.bar_density]
    print(shown.pivot(index="dataset", columns="network", values="transient_spearman").to_string())
    print(f"Wrote results to {outdir}")


if __name__ == "__main__":
    main()
