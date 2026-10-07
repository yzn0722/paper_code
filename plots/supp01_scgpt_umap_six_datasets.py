#!/usr/bin/env python3
"""Six dataset-specific joint UMAPs, following the mHSC-L all-cell workflow.

Each panel fits its own scaler, PCA, and UMAP to its observed early,
intermediate, late, and scGPT-predicted late-like cells. Coordinates must not
be compared across panels as a shared embedding.
"""

from __future__ import annotations

import gc
import hashlib
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path("/mnt/10T/yzn/benchmark_GRN")
SCGPT_REPO = ROOT / "pre_scgpt/scGPT"
MODEL_DIR = SCGPT_REPO / "scgpt_human"
SOURCE_DIR = ROOT / "pre_scgpt/0204code/Pseudotime_trajectory_plots_umap"
OUTDIR = SOURCE_DIR / "six_dataset_joint_umap_binned_ema09"
OUTPUT = OUTDIR / "scGPT_allcells_joint_umap_six_datasets_2x3"
DATASETS = ("hESC", "hHep", "mHSC-E", "mHSC-GM", "mHSC-L", "mDC")
PT_QUANTILE = 0.2
GEN_ITERS = 16
BATCH_SIZE = 16
EMA_ALPHA = 0.9  # Retention: 0.9 * previous state + 0.1 * reconstruction.
SEED = 42
COLORS = {
    "early": "#E69F00",
    "late": "#4EA3F1",
    "pred": "#FF9A3D",
    "middle": "#9B9B9B",
}

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "upstream" / "dynamics"))
from run_multimodel_pseudotime import bin_expr_to_0_50 as native_bin_expr


def bin_expr_to_0_50(x: np.ndarray, n_bins: int = 51) -> np.ndarray:
    """Native per-cell scGPT quantile bins, with log1p disabled as in Methods."""
    return native_bin_expr(x, do_log1p=False, n_bins=n_bins)


def load_input_config():
    sys.path.insert(0, str(SCGPT_REPO))
    from scgpt.tokenizer.gene_tokenizer import GeneVocab

    cfg = json.loads((MODEL_DIR / "args.json").read_text(encoding="utf-8"))
    vocab = GeneVocab.from_file(MODEL_DIR / "vocab.json")
    return vocab, int(cfg.get("n_bins", 51))


def load_model(device: torch.device):
    from types import SimpleNamespace
    from run_multimodel_pseudotime import load_scgpt_model

    return load_scgpt_model(SimpleNamespace(scgpt_model_dir=str(MODEL_DIR),
                                          scgpt_repo_dir=str(SCGPT_REPO)), device)


@torch.no_grad()
def generate_per_cell(
    model, vocab, x_early: np.ndarray,
    genes: list[str], device: torch.device, dataset: str,
) -> np.ndarray:
    gene_names = [gene.upper() for gene in genes]
    gene_ids = np.array([vocab[g] if g in vocab else vocab["<pad>"] for g in gene_names])
    gene_ids = np.concatenate([[vocab["<cls>"]], gene_ids])
    gene_ids_tensor = torch.tensor(gene_ids[None, :], dtype=torch.long)
    x_in = np.concatenate([np.zeros((len(x_early), 1), dtype=np.float32), x_early], axis=1)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    vals_all = torch.tensor(x_in, dtype=dtype)
    pad_mask = gene_ids_tensor.eq(vocab["<pad>"]).expand(len(x_early), -1)
    update_mask = torch.ones((1, len(gene_ids)), dtype=torch.bool, device=device)
    update_mask[:, 0] = False

    for iteration in range(GEN_ITERS):
        for start in range(0, len(vals_all), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(vals_all))
            batch_size = end - start
            vals = vals_all[start:end].to(device)
            src = gene_ids_tensor.expand(batch_size, -1).to(device)
            mask = pad_mask[start:end].to(device)
            freeze = mask | (~update_mask.expand(batch_size, -1))
            predicted = model(src=src, values=vals, src_key_padding_mask=mask)["mlm_output"]
            vals = torch.where(freeze, vals, EMA_ALPHA * vals + (1.0 - EMA_ALPHA) * predicted)
            vals_all[start:end] = vals.detach().cpu()
        print(f"{dataset}: iteration {iteration + 1}/{GEN_ITERS}", flush=True)
    return vals_all[:, 1:].float().numpy()


def load_dataset(dataset: str, vocab, n_bins: int = 51) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
    expr_path = ROOT / "input_process/CHIP" / f"{dataset}_chip_matched-ExpressionData.csv"
    pt_path = ROOT / "PseudoTime" / dataset / "PseudoTime.csv"
    expr = pd.read_csv(expr_path, index_col=0)
    pt_frame = pd.read_csv(pt_path)
    pt_frame = pt_frame.rename(
        columns={pt_frame.columns[0]: "cell", pt_frame.columns[1]: "pt"}
    ).set_index("cell")
    common = expr.columns.intersection(pt_frame.index)
    if not len(common):
        raise ValueError(f"{dataset}: no shared expression/pseudotime cell IDs")
    expr = expr[common]
    pt = pd.to_numeric(pt_frame.loc[common, "pt"], errors="coerce").to_numpy(float)
    valid = np.isfinite(pt)
    invalid_count = int((~valid).sum())
    expr, pt = expr.loc[:, valid], pt[valid]
    if len(pt) < 10:
        raise ValueError(f"{dataset}: too few valid pseudotime cells")
    genes = expr.index.astype(str).tolist()
    mapped = np.array([gene.upper() in vocab for gene in genes], dtype=bool)
    if not mapped.any():
        raise ValueError(f"{dataset}: no genes mapped to scGPT vocabulary")
    raw = expr.T.to_numpy(dtype=np.float32)
    x_all = np.zeros_like(raw, dtype=np.float32)
    x_all[:, mapped] = bin_expr_to_0_50(raw[:, mapped], n_bins=n_bins)
    lo, hi = np.quantile(pt, [PT_QUANTILE, 1.0 - PT_QUANTILE])
    early_mask, late_mask = pt <= lo, pt >= hi
    if np.any(early_mask & late_mask):
        raise ValueError(f"{dataset}: early and late masks overlap")
    middle_mask = ~(early_mask | late_mask)
    summary = {
        "dataset": dataset, "input_cells": int(len(common)),
        "invalid_pseudotime_cells": invalid_count, "valid_cells": int(len(pt)),
        "genes": len(genes), "early": int(early_mask.sum()),
        "middle": int(middle_mask.sum()), "late": int(late_mask.sum()),
        "quantile": PT_QUANTILE, "early_threshold": float(lo),
        "late_threshold": float(hi),
        "mapped_genes": int(mapped.sum()), "value_preprocessing": "scgpt.preprocess.binning",
        "n_bins": n_bins, "log1p": False, "ema_alpha": EMA_ALPHA,
        "ema_definition": "alpha * previous + (1-alpha) * prediction",
    }
    return x_all[early_mask], x_all[middle_mask], x_all[late_mask], genes, summary


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prediction_identity(x_early, genes, state):
    if "model_file_hashes" not in state:
        state["model_file_hashes"] = {
            name: _file_sha256(MODEL_DIR / name)
            for name in ("args.json", "vocab.json", "best_model.pt")
        }
    return {
        "protocol_version": 1, "ema_alpha": EMA_ALPHA, "gen_iters": GEN_ITERS,
        "batch_size": BATCH_SIZE, "seed": SEED,
        "log1p": False, "n_bins": state["n_bins"],
        "value_preprocessing": "native_per_cell_bins_of_mapped_genes",
        "input_shape": list(x_early.shape),
        "input_sha256": hashlib.sha256(np.ascontiguousarray(x_early, dtype=np.float32).tobytes()).hexdigest(),
        "genes": genes, "model_files": state["model_file_hashes"],
        "source_sha256": _file_sha256(Path(__file__)),
    }


def get_prediction(
    dataset: str, x_early: np.ndarray, genes: list[str],
    device: torch.device, state: dict,
) -> np.ndarray:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / f"{dataset}_scgpt_early_to_latelike_predicted_cells.npy"
    manifest_path = path.with_suffix(".json")
    identity = prediction_identity(x_early, genes, state)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    reusable = (path.exists() and manifest.get("identity") == identity
                and manifest.get("prediction_sha256") == _file_sha256(path))
    source = "validated current-protocol cache" if reusable else "new current-protocol prediction"
    if reusable:
        x_pred = np.load(path).astype(np.float32)
    else:
        if "model" not in state:
            state["model"], state["vocab"] = load_model(device)
        x_pred = generate_per_cell(state["model"], state["vocab"], x_early, genes, device, dataset)
    if x_pred.shape != x_early.shape or not np.isfinite(x_pred).all():
        raise ValueError(f"{dataset}: invalid prediction shape or nonfinite values: {x_pred.shape}")
    if not reusable:
        np.save(path, x_pred)
        manifest_path.write_text(json.dumps({"identity": identity,
                                            "prediction_sha256": _file_sha256(path)}, indent=2), encoding="utf-8")
    print(f"{dataset}: {source}; prediction shape={x_pred.shape}", flush=True)
    return x_pred


def embed_dataset(dataset: str, device: torch.device, state: dict) -> None:
    import umap

    # Native binning may randomize quantile ties. Reset independently of cache hits.
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    cache = OUTDIR / f"{dataset}_joint_umap_coordinates.npz"
    if "vocab" not in state:
        state["vocab"], state["n_bins"] = load_input_config()
    x_early, x_middle, x_late, genes, summary = load_dataset(dataset, state["vocab"], state["n_bins"])
    x_pred = get_prediction(dataset, x_early, genes, device, state)
    combined = np.vstack([x_early, x_middle, x_late, x_pred]).astype(np.float32)
    scaled = StandardScaler().fit_transform(combined)
    n_pcs = min(50, scaled.shape[0] - 1, scaled.shape[1])
    pcs = PCA(n_components=n_pcs, random_state=SEED).fit_transform(scaled)
    embedding = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2,
        metric="euclidean", random_state=SEED, n_jobs=1,
    ).fit_transform(pcs)
    n_early, n_middle, n_late = len(x_early), len(x_middle), len(x_late)
    i1, i2, i3 = n_early, n_early + n_middle, n_early + n_middle + n_late
    middle = embedding[i1:i2].copy()
    clouds = {
        "early": embedding[:i1].copy(),
        "late": embedding[i2:i3].copy(),
        "pred": embedding[i3:].copy(),
    }
    np.savez_compressed(cache, early=clouds["early"], middle=middle,
                        late=clouds["late"], pred=clouds["pred"])
    with (OUTDIR / f"{dataset}_data_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"{dataset}: cells={summary['valid_cells']} genes={summary['genes']} "
          f"early={n_early} middle={n_middle} late={n_late} pred={len(x_pred)}", flush=True)


def plot_panel(ax: plt.Axes, dataset: str) -> None:
    with np.load(OUTDIR / f"{dataset}_joint_umap_coordinates.npz") as cached:
        clouds = {label: cached[label].copy() for label in ("early", "middle", "late", "pred")}
    centers = {label: clouds[label].mean(axis=0) for label in ("early", "late", "pred")}
    middle = clouds["middle"]
    ax.scatter(middle[:, 0], middle[:, 1], s=19, alpha=0.38,
               color=COLORS["middle"], edgecolors="none", rasterized=True, zorder=0)
    start = centers["early"]
    for target_label, curvature in (("late", -0.16), ("pred", 0.16)):
        target_cloud = clouds[target_label]
        # Terminate at an observed target cell rather than a projected radius;
        # elongated/disconnected clouds can otherwise leave the arrow in space.
        target_edge = target_cloud[np.argmin(np.linalg.norm(target_cloud - start, axis=1))]
        ax.add_patch(FancyArrowPatch(
            start, target_edge, connectionstyle=f"arc3,rad={curvature}",
            arrowstyle="-|>", mutation_scale=15, linewidth=2.3,
            color=COLORS[target_label], alpha=0.88,
            shrinkA=12, shrinkB=4, zorder=1,
        ))
    for label in ("early", "late", "pred"):
        cloud = clouds[label]
        ax.scatter(cloud[:, 0], cloud[:, 1], s=34, alpha=0.78,
                   color=COLORS[label], edgecolors="none", rasterized=True, zorder=2)
        center = centers[label]
        ax.scatter(center[0], center[1], s=135, marker="X", color=COLORS[label],
                   edgecolors="white", linewidths=1.0, zorder=5)
    ax.set_title(dataset, fontsize=16, pad=8, loc="left")
    ax.set_xlabel("UMAP 1", fontsize=16)
    ax.set_ylabel("UMAP 2", fontsize=16)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis="both", which="both", labelsize=14, length=0)
    ax.margins(x=0.025, y=0.025)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)


def check_panel_alignment(fig, axes) -> None:
    """Check actual 2-by-3 axes bounds without an external audit-module import."""
    width, height = fig.get_size_inches() * 72
    boxes = np.array([ax.get_position().bounds for ax in axes.flat]).reshape(2, 3, 4)
    boxes *= np.array([width, height, width, height])
    checks = {
        "equal_width": float(np.ptp(boxes[:, :, 2])),
        "equal_height": float(np.ptp(boxes[:, :, 3])),
        "row_alignment": float(max(np.ptp(row[:, 1]) for row in boxes)),
        "column_alignment": float(max(np.ptp(boxes[:, j, 0]) for j in range(3))),
    }
    report = {"axes_bounds_pt": boxes.tolist(), "deviations_pt": checks,
              "tolerance_pt": 1.5, "pass": all(v <= 1.5 for v in checks.values())}
    Path(str(OUTPUT) + ".alignment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["pass"]:
        raise RuntimeError(f"UMAP panels are misaligned: {checks}")


def plot_composite() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none", "axes.unicode_minus": False,
    })
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=300)
    for ax, dataset in zip(axes.flat, DATASETS):
        plot_panel(ax, dataset)
    handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=8,
               color=COLORS[label], label=title)
        for label, title in (
            ("middle", "Observed Intermediate cells"),
            ("early", "Observed Early cells"),
            ("late", "Observed Late cells"),
            ("pred", "Predicted Late cells"),
        )
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985),
               ncol=4, frameon=False, fontsize=14, handletextpad=0.35,
               columnspacing=1.6)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.075,
                        top=0.89, wspace=0.23, hspace=0.34)
    fig.canvas.draw()
    check_panel_alignment(fig, axes)
    fig.savefig(OUTPUT.with_suffix(".pdf"), dpi=300)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"saved={OUTPUT.with_suffix('.pdf')}", flush=True)


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU is unavailable; refusing to run slow CPU scGPT inference")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    state: dict = {}
    print(f"device={device}; datasets={','.join(DATASETS)}", flush=True)
    for dataset in DATASETS:
        embed_dataset(dataset, device, state)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    plot_composite()


if __name__ == "__main__":
    main()
