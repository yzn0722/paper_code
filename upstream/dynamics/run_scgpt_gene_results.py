import os
import re
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
os.environ["KMP_WARNINGS"] = "off"

# =====================================================
# 固定配置
# =====================================================
MODEL_DIR = "/mnt/10T/yzn/benchmark_GRN/pre_scgpt/scGPT/scgpt_human"
OUTDIR = "results_multidataset_pseudotime_0326"

# =====================================================
# 多数据集配置
# =====================================================
DATASETS = {
     "hESC": {
        "expr_csv": "/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/hESC_chip_matched-ExpressionData.csv",
        "pt_csv":   "/mnt/10T/yzn/benchmark_GRN/PseudoTime/hESC/PseudoTime.csv",
        "species": "human",
    },
    "hHep": {
        "expr_csv": "/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/hHep_chip_matched-ExpressionData.csv",
        "pt_csv":   "/mnt/10T/yzn/benchmark_GRN/PseudoTime/hHep/PseudoTime.csv",
        "species": "human",
    },
    "mDC": {
        "expr_csv": "/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/mDC_chip_matched-ExpressionData.csv",
        "pt_csv":   "/mnt/10T/yzn/benchmark_GRN/PseudoTime/mDC/PseudoTime.csv",
        "species": "mouse",
    },
    "mHSC-E": {
        "expr_csv": "/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/mHSC-E_chip_matched-ExpressionData.csv",
        "pt_csv":   "/mnt/10T/yzn/benchmark_GRN/PseudoTime/mHSC-E/PseudoTime.csv",
        "species": "mouse",
    },
    "mHSC-GM": {
        "expr_csv": "/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/mHSC-GM_chip_matched-ExpressionData.csv",
        "pt_csv":   "/mnt/10T/yzn/benchmark_GRN/PseudoTime/mHSC-GM/PseudoTime.csv",
        "species": "mouse",
    },
    "mHSC-L": {
        "expr_csv": "/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/mHSC-L_chip_matched-ExpressionData.csv",
        "pt_csv":   "/mnt/10T/yzn/benchmark_GRN/PseudoTime/mHSC-L/PseudoTime.csv",
        "species": "mouse",
    },
}

# =====================================================
# 关键参数（核心修改：TOP_PERCENT替代TOPK）
# =====================================================
PT_QUANTILE = 0.2
TOP_PERCENT = 30  # 每个数据集评估「拟时间变化幅度最大的前N%基因」
GEN_ITERS = 16
BATCH_SIZE = 16  # 显存不足可调小（如16/8）
EMA_ALPHA = 0.9  # paper α: x <- α x + (1-α) x_hat (retention on previous state)
NO_LOG1P = True
# Align with scFoundation: only score mapped genes; near-zero deltas are not Up/Down.
EPS_DIR = 1e-3

# =====================================================
# scGPT
# =====================================================
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pseudotime_utils import load_aligned_pseudotime, split_pseudotime_quantiles

sys.path.insert(0, "/mnt/10T/yzn/benchmark_GRN/pre_scgpt/scGPT")
from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import binning as scgpt_binning


# =====================================================
# Utils
# =====================================================
def bin_expr_to_0_50(x, do_log1p=True, n_bins=51):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] == 0 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("scGPT binning requires a nonempty, finite, non-negative cell-by-gene matrix")
    if do_log1p:
        x = np.log1p(x)
    return np.stack([scgpt_binning(row, n_bins=n_bins) for row in x]).astype(np.float32)


def convert_mouse_to_human_gene(gene_name):
    """
    小鼠基因名转人类基因名的简单规则：
    小鼠: 首字母大写其余小写 (e.g., Gapdh)
    人类: 全部大写 (e.g., GAPDH)
    """
    return gene_name.upper()


def direction_label(delta, eps=EPS_DIR):
    """Return Up / Down / '' (near-zero excluded from scoring)."""
    if delta > eps:
        return "Up"
    if delta < -eps:
        return "Down"
    return ""


def balanced_direction_accuracy(pred_delta, true_delta, eval_idx, eps=EPS_DIR):
    """Balanced Up/Down recall on eval_idx; near-zero true deltas are dropped."""
    td = true_delta[eval_idx]
    pd = pred_delta[eval_idx]
    true_dir = np.where(td > eps, 1, np.where(td < -eps, -1, 0))
    pred_dir = np.where(pd > eps, 1, np.where(pd < -eps, -1, 0))
    valid = true_dir != 0
    recalls = []
    for cls in (-1, 1):
        m = valid & (true_dir == cls)
        if m.any():
            recalls.append(float((pred_dir[m] == cls).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


# =====================================================
# Build model
# =====================================================
def build_model(model_dir, device):
    # Also works when this evaluator is imported by file path by the seeded runner.
    helper_dir = str(Path(__file__).resolve().parent)
    if helper_dir not in sys.path:
        sys.path.insert(0, helper_dir)
    from scgpt_checkpoint import load_scgpt_dynamics_checkpoint

    with open(Path(model_dir) / "args.json") as f:
        cfg = json.load(f)

    vocab = GeneVocab.from_file(Path(model_dir) / "vocab.json")
    for t in ["<pad>", "<cls>", "<eoc>"]:
        if t not in vocab:
            vocab.append_token(t)

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=cfg["embsize"],
        nhead=cfg["nheads"],
        d_hid=cfg["d_hid"],
        nlayers=cfg["nlayers"],
        vocab=vocab,
        pad_value=cfg["pad_value"],
        n_input_bins=cfg.get("n_bins", 51),
        use_fast_transformer=cfg.get("fast_transformer", True),
    )

    ckpt = torch.load(Path(model_dir) / "best_model.pt", map_location="cpu")
    model.checkpoint_load_report = load_scgpt_dynamics_checkpoint(
        model, ckpt, checkpoint_path=Path(model_dir) / "best_model.pt"
    )
    model.to(device)
    model.eval()

    if device.type == "cuda":
        model.half()

    return model, vocab


# =====================================================
# Iterative generation + accuracy curve
# =====================================================
@torch.no_grad()
def iterative_direction_accuracy(
    model,
    gene_ids_tensor,
    values_tensor,
    pad_mask,
    update_mask_1d,
    early_mean,
    true_delta,
    top_idx,
):
    device = next(model.parameters()).device
    vals_all = values_tensor.clone()
    update_mask = torch.tensor(update_mask_1d[None, :], device=device).bool()

    acc_curve = []

    for it in range(GEN_ITERS):
        for start in range(0, vals_all.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, vals_all.shape[0])
            bs = end - start

            vals = vals_all[start:end].to(device)
            src = gene_ids_tensor.expand(bs, -1).to(device)
            mask = pad_mask[start:end].to(device)

            freeze = mask | (~update_mask.expand(bs, -1))
            out = model(src=src, values=vals, src_key_padding_mask=mask)
            new_vals = out["mlm_output"]

            # Paper: x <- α x + (1-α) x_hat with α=EMA_ALPHA (default 0.9).
            vals = torch.where(
                freeze,
                vals,
                EMA_ALPHA * vals + (1 - EMA_ALPHA) * new_vals
            )

            vals_all[start:end] = vals.detach().cpu()

        # Balanced direction accuracy on mapped top-% genes (EPS_DIR; zero ≠ Down).
        pred_mean = vals_all[:, 1:].numpy().mean(axis=0)
        pred_delta = pred_mean - early_mean
        acc = balanced_direction_accuracy(pred_delta, true_delta, top_idx)
        acc_curve.append(acc)

    # 返回最终迭代后的预测均值
    pred_late_like_mean = vals_all[:, 1:].numpy().mean(axis=0)
    return acc_curve, pred_late_like_mean


# =====================================================
# Run one dataset + 导出CSV结果
# =====================================================
def run_dataset(name, cfg, model, vocab, device, outdir):
    print(f"\n{'='*60}")
    print(f"Running {name}")
    print(f"{'='*60}")

    expr = pd.read_csv(cfg["expr_csv"], index_col=0)
    expr, pt, pt_stats = load_aligned_pseudotime(expr, cfg["pt_csv"])
    early, late, lo, hi = split_pseudotime_quantiles(pt, PT_QUANTILE)
    print(f"  [INFO] Dropped invalid matched pseudotime cells: {pt_stats['invalid_matched_pseudotime_cells']}")

    # 获取原始基因名；统一大写以匹配 scGPT human vocab
    genes_original = expr.index.astype(str).tolist()
    genes = [convert_mouse_to_human_gene(g) for g in genes_original]
    species = cfg.get("species", "human")
    if species == "mouse":
        print(f"  [INFO] Mouse data detected, converting gene names to uppercase")

    is_mapped = np.array([g in vocab for g in genes], dtype=bool)
    matched = int(is_mapped.sum())
    match_rate = matched / max(len(genes), 1) * 100
    print(f"  [INFO] Total genes: {len(genes)}")
    print(f"  [INFO] Vocab matched: {matched} ({match_rate:.1f}%)")
    print(f"  [INFO] Total valid cells: {len(pt)}")

    X = expr.T.to_numpy(dtype=np.float32)
    with open(Path(MODEL_DIR) / "args.json") as f:
        n_bins = int(json.load(f).get("n_bins", 51))
    if not is_mapped.any():
        raise ValueError(f"{name}: no genes mapped to scGPT vocab")
    X_bin = np.zeros_like(X, dtype=np.float32)
    X_bin[:, is_mapped] = bin_expr_to_0_50(
        X[:, is_mapped], do_log1p=(not NO_LOG1P), n_bins=n_bins
    )

    print(f"  [INFO] Early cells (pt <= {lo:.3f}): {early.sum()}")
    print(f"  [INFO] Late cells (pt >= {hi:.3f}): {late.sum()}")

    early_mean = X_bin[early].mean(axis=0)
    late_mean = X_bin[late].mean(axis=0)
    true_delta = late_mean - early_mean

    # Top-% among mapped genes only (same policy as scFoundation).
    mapped_pool = np.where(is_mapped)[0]
    if len(mapped_pool) == 0:
        raise ValueError(f"{name}: no genes mapped to scGPT vocab")
    top_n = max(int(len(mapped_pool) * TOP_PERCENT / 100), 1)
    order = np.argsort(np.abs(true_delta[mapped_pool]))[::-1]
    top_idx = mapped_pool[order[:top_n]].copy()
    top_match_rate = 100.0
    print(
        f"  [INFO] Top-{TOP_PERCENT}% of mapped genes for evaluation: "
        f"{top_n}/{len(mapped_pool)} (EPS_DIR={EPS_DIR})"
    )

    gene_ids = np.array([vocab[g] if g in vocab else vocab["<pad>"] for g in genes])
    gene_ids = np.concatenate([[vocab["<cls>"]], gene_ids])
    gene_ids_tensor = torch.tensor(gene_ids[None, :], dtype=torch.long)

    X_in = np.concatenate([np.zeros((X_bin.shape[0], 1)), X_bin], axis=1)
    pad_mask = gene_ids_tensor.eq(vocab["<pad>"]).expand(X_in.shape[0], -1)
    values_tensor = torch.tensor(X_in, dtype=torch.float16 if device.type == "cuda" else torch.float32)

    update_mask_1d = np.zeros(gene_ids_tensor.shape[1], dtype=bool)
    update_mask_1d[1:] = True
    # Freeze OOV pads via pad_mask; only mapped positions can change.
    print(f"  [INFO] Iteration covers mapped genes only via pad freeze ({matched}/{len(genes)})")

    # 运行迭代，获取预测结果
    acc_curve, pred_late_like_mean = iterative_direction_accuracy(
        model,
        gene_ids_tensor,
        values_tensor[early],
        pad_mask[early],
        update_mask_1d,
        early_mean,
        true_delta,
        top_idx,
    )
    
    print(f"  [RESULT] Final balanced accuracy (Top-{TOP_PERCENT}% mapped): {acc_curve[-1]:.2%}")

    # ==============================================
    # 导出 CSV：近零方向留空；OOV 不参与 dir_correct
    # ==============================================
    pred_delta = pred_late_like_mean - early_mean

    dir_true = [direction_label(d) for d in true_delta]
    dir_pred = [direction_label(d) for d in pred_delta]
    in_eval = np.zeros(len(genes), dtype=bool)
    in_eval[top_idx] = True
    dir_correct = [
        (1 if dt == dp else 0) if (dt != "" and dp != "") else 0
        for dt, dp in zip(dir_true, dir_pred)
    ]

    result_df = pd.DataFrame({
        "gene": genes_original,
        "gene_used": genes,
        "is_mapped": is_mapped.astype(int),
        "in_eval": in_eval.astype(int),
        "true_early_mean": early_mean,
        "true_late_mean": late_mean,
        "pred_late_like_mean": pred_late_like_mean,
        "delta_true": true_delta,
        "delta_pred": pred_delta,
        "dir_true": dir_true,
        "dir_pred": dir_pred,
        "dir_correct": dir_correct,
    })

    csv_path = outdir / f"{name}_gene_result.csv"
    result_df.to_csv(csv_path, index=False, float_format="%.15g")
    print(f"  [OK] 结果已导出：{csv_path}")

    # 诊断信息
    diagnostics = {
        "n_genes": len(genes),
        "n_cells": len(pt),
        "pseudotime_filter": pt_stats,
        "vocab_match_rate": match_rate,
        "top_vocab_match_rate": top_match_rate,
        "n_early": int(early.sum()),
        "n_late": int(late.sum()),
        "iter_genes_count": int(matched),
        "eval_percent": TOP_PERCENT,
        "eval_genes_count": int(top_n),
        "eps_dir": EPS_DIR,
        "eval_policy": "top_percent_of_mapped_genes; EPS_DIR excludes near-zero",
    }
    
    return acc_curve, diagnostics


# =====================================================
# Nature-style plotting
# =====================================================
def plot_nature_style(all_curves, diagnostics, outdir):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 8,
        'axes.linewidth': 0.8,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'legend.fontsize': 8,
        'legend.frameon': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })
    
    colors = {
        'hHep': '#E64B35',
        'mDC': '#4DBBD5',
        'mHSC-E': '#00A087',
        'mHSC-GM': '#3C5488',
        'mHSC-L': '#F39B7F',
        'hESC': '#8491B4',
    }
    
    # 收敛曲线
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for name, acc in all_curves.items():
        ax.plot(range(1, GEN_ITERS + 1), acc, marker='o', markersize=4, linewidth=1.5,
                color=colors.get(name, '#666666'), label=name)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(f'Direction accuracy (Top-{TOP_PERCENT}% genes)')
    ax.set_xlim(0.5, GEN_ITERS + 0.5)
    ax.set_ylim(0.4, 1.0)
    ax.set_xticks(range(1, GEN_ITERS + 1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right', ncol=2)
    plt.tight_layout()
    plt.savefig(outdir / "fig1_convergence.pdf")
    plt.savefig(outdir / "fig1_convergence.png")
    plt.close()
    
    # 最终准确率柱状图
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    names = list(all_curves.keys())
    final_acc = [all_curves[n][-1] for n in names]
    bar_colors = [colors.get(n, '#666666') for n in names]
    bars = ax.bar(range(len(names)), final_acc, color=bar_colors, width=0.6, edgecolor='black', linewidth=0.5)
    for i, (bar, v) in enumerate(zip(bars, final_acc)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{v:.1%}', ha='center', va='bottom', fontsize=7)
    ax.set_ylabel(f'Direction accuracy (Top-{TOP_PERCENT}% genes)')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(outdir / "fig2_final_accuracy.pdf")
    plt.savefig(outdir / "fig2_final_accuracy.png")
    plt.close()
    
    # Vocab匹配率 vs 准确率
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    for name in names:
        match_rate = diagnostics[name]['top_vocab_match_rate']
        acc = all_curves[name][-1]
        ax.scatter(match_rate, acc, s=60, c=colors.get(name, '#666666'), 
                   edgecolor='black', linewidth=0.5, label=name, zorder=3)
    ax.set_xlabel(f'Top-{TOP_PERCENT}% genes vocab match rate (%)')
    ax.set_ylabel(f'Direction accuracy (Top-{TOP_PERCENT}% genes)')
    ax.set_xlim(0, 105)
    ax.set_ylim(0.4, 1.0)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right', fontsize=7)
    plt.tight_layout()
    plt.savefig(outdir / "fig3_vocab_vs_accuracy.pdf")
    plt.savefig(outdir / "fig3_vocab_vs_accuracy.png")
    plt.close()
    
    # 组合图
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
    ax = axes[0]
    for name, acc in all_curves.items():
        ax.plot(range(1, GEN_ITERS + 1), acc, marker='o', markersize=4, linewidth=1.5,
                color=colors.get(name, '#666666'), label=name)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(f'Direction accuracy (Top-{TOP_PERCENT}% genes)')
    ax.set_xlim(0.5, GEN_ITERS + 0.5)
    ax.set_ylim(0.4, 1.0)
    ax.set_xticks(range(1, GEN_ITERS + 1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right', fontsize=7)
    ax.text(-0.15, 1.05, 'a', transform=ax.transAxes, fontsize=12, fontweight='bold')
    
    ax = axes[1]
    bars = ax.bar(range(len(names)), final_acc, color=bar_colors, width=0.6, edgecolor='black', linewidth=0.5)
    for i, (bar, v) in enumerate(zip(bars, final_acc)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{v:.1%}', ha='center', va='bottom', fontsize=7)
    ax.set_ylabel(f'Direction accuracy (Top-{TOP_PERCENT}% genes)')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.15, 1.05, 'b', transform=ax.transAxes, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outdir / "fig_combined.pdf")
    plt.savefig(outdir / "fig_combined.png")
    plt.close()
    
    print(f"\n[OK] Nature-style figures saved to: {outdir}")


# =====================================================
# Main
# =====================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model, vocab = build_model(MODEL_DIR, device)
    print(f"Vocab size: {len(vocab)}")
    print(f"Run mode: Iterate ALL genes, Evaluate Top-{TOP_PERCENT}% genes")

    all_curves = {}
    all_diagnostics = {}
    outdir = Path(OUTDIR)
    outdir.mkdir(exist_ok=True, parents=True)

    for name, cfg in DATASETS.items():
        acc_curve, diag = run_dataset(name, cfg, model, vocab, device, outdir)
        all_curves[name] = acc_curve
        all_diagnostics[name] = diag

    # 保存结果
    with open(outdir / "accuracy_curves.json", "w") as f:
        json.dump(all_curves, f, indent=2)
    
    with open(outdir / "diagnostics.json", "w") as f:
        json.dump(all_diagnostics, f, indent=2)

    # 绘图
    plot_nature_style(all_curves, all_diagnostics, outdir)
    
    # 打印汇总
    print("\n" + "="*80)
    print(f"SUMMARY (Top-{TOP_PERCENT}% genes evaluation)")
    print("="*80)
    print(f"{'Dataset':<12} {'Total Genes':<12} {'Eval Genes':<12} {'Vocab%':<10} {'Top% Vocab%':<12} {'Accuracy':<10}")
    print("-"*80)
    for name in all_curves:
        d = all_diagnostics[name]
        acc = all_curves[name][-1]
        print(f"{name:<12} {d['n_genes']:<12} {d['eval_genes_count']:<12} {d['vocab_match_rate']:<10.1f} {d['top_vocab_match_rate']:<12.1f} {acc:<10.2%}")
    print("="*80)


if __name__ == "__main__":
    main()
