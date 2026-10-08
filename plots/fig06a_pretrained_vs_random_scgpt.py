#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制 Random 和 Weight 两个模型在 top30 数据集上的平衡准确率对比柱状图
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# Match the reference convergence figure's font family and text sizes.
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "none"})

# Match the 8 x 6 inch plotting canvas used by weight.py.
FIGURE_SIZE_IN = (8, 6)
DEFAULT_RANDOM_RESULTS_DIR = Path(
    "/mnt/10T/yzn/paper-code/outputs/fig06a_native_binning_ema09_20261008/random"
)
DEFAULT_WEIGHT_RESULTS_DIR = Path(
    "/mnt/10T/yzn/paper-code/outputs/fig06a_native_binning_ema09_20261008/pretrained"
)
DEFAULT_OUTPUT_PDF = Path(__file__).resolve().with_name(
    "top30_balanced_accuracy_seeded10_mapped_top30.pdf"
)
DEFAULT_VOCAB_PATH = Path(
    "/mnt/10T/yzn/benchmark_GRN/pre_scgpt/scGPT/scgpt_human/vocab.json"
)
EXPECTED_SEEDS = tuple(range(1, 11))
EPS_DIR = 1e-3  # Keep aligned with upstream/dynamics/run_scgpt_gene_results.py.

def load_vocab_tokens(vocab_path):
    """Load the scGPT vocabulary tokens used to identify mapped genes."""
    payload = json.loads(vocab_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return {str(token).strip().upper() for token in payload}
    if isinstance(payload, list):
        return {str(token).strip().upper() for token in payload}
    raise ValueError(f"Unsupported vocabulary format: {vocab_path}")


def calculate_metrics(file_path, dataset, vocab_tokens, use_top30=True, top_percent=0.3):
    """Calculate BA using mapped top-% genes and the evaluator EPS_DIR policy."""
    df = pd.read_csv(file_path)

    required_cols = ["delta_true", "delta_pred"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"{file_path} is missing required columns: {required_cols}")
    true_delta = pd.to_numeric(df["delta_true"], errors="raise").to_numpy(dtype=float)
    pred_delta = pd.to_numeric(df["delta_pred"], errors="raise").to_numpy(dtype=float)

    if "in_eval" in df.columns:
        in_eval = pd.to_numeric(df["in_eval"], errors="raise").to_numpy()
        eval_idx = np.flatnonzero(in_eval == 1)
    elif use_top30:
        if "is_mapped" in df.columns:
            mapped_values = df["is_mapped"]
            if mapped_values.dtype == bool:
                mapped = mapped_values.to_numpy()
            else:
                mapped = pd.to_numeric(mapped_values, errors="raise").to_numpy() == 1
        else:
            gene_col = "gene_used" if "gene_used" in df.columns else "gene"
            if gene_col not in df.columns or vocab_tokens is None:
                raise ValueError(
                    f"{file_path} lacks in_eval/is_mapped; a gene column and vocab are required"
                )
            gene_names = df[gene_col].fillna("").astype(str).str.strip().str.upper()
            mapped = gene_names.isin(vocab_tokens).to_numpy()
        mapped_pool = np.flatnonzero(mapped)
        if mapped_pool.size == 0:
            raise ValueError(f"{file_path} has no genes mapped to the supplied vocabulary")
        n_top = max(int(mapped_pool.size * top_percent), 1)
        order = np.argsort(np.abs(true_delta[mapped_pool]))[::-1]
        eval_idx = mapped_pool[order[:n_top]]
    else:
        eval_idx = np.arange(len(df))

    if eval_idx.size == 0:
        return None
    true_eval = true_delta[eval_idx]
    pred_eval = pred_delta[eval_idx]
    true_dir = np.where(true_eval > EPS_DIR, 1, np.where(true_eval < -EPS_DIR, -1, 0))
    pred_dir = np.where(pred_eval > EPS_DIR, 1, np.where(pred_eval < -EPS_DIR, -1, 0))
    valid_true = true_dir != 0
    recalls = []
    for label in (-1, 1):
        class_rows = valid_true & (true_dir == label)
        if class_rows.any():
            recalls.append(float(np.mean(pred_dir[class_rows] == label)))
    return float(np.mean(recalls)) if recalls else None

def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_matched_conditions(random_dir, pretrained_dir, datasets, result_files):
    """Require the same recorded protocol, observed deltas and scoring gene set."""
    random_manifest = json.loads((random_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    pretrained_manifest = json.loads((pretrained_dir / "seed_manifest.json").read_text(encoding="utf-8"))
    protocol = random_manifest.get("protocol")
    if not protocol or protocol != pretrained_manifest.get("protocol"):
        raise ValueError("Pretrained and random results lack an identical recorded protocol; rerun both")
    if protocol.get("ema_alpha") != .9 or protocol.get("input_processing") != "scgpt.preprocess.binning per cell on mapped genes":
        raise ValueError("Fig. 6a requires native per-cell binning and EMA retention 0.9")
    required = ["gene", "gene_used", "is_mapped", "in_eval", "true_early_mean", "true_late_mean", "delta_true"]
    for dataset in datasets:
        path = pretrained_dir / f"{dataset}_gene_result.csv"
        expected = pretrained_manifest.get("metrics", {}).get(dataset, {}).get("csv_sha256")
        if not expected or _sha256_file(path) != expected:
            raise ValueError(f"Pretrained result hash mismatch: {path}")
        observed = pd.read_csv(path)[required]
        for random_path in result_files[dataset]:
            seed_manifest = json.loads((random_path.parent / "seed_manifest.json").read_text(encoding="utf-8"))
            if seed_manifest.get("protocol") != protocol:
                raise ValueError(f"Seed protocol differs: {random_path}")
            try:
                pd.testing.assert_frame_equal(observed, pd.read_csv(random_path)[required],
                                              check_exact=True)
            except AssertionError as exc:
                raise ValueError(f"Observed input or evaluation genes differ: {random_path}") from exc
    print("[OK] All 50 random results match pretrained observed inputs and evaluation genes exactly")


def validate_random_results(results_dir, datasets):
    """Require the recorded ten-seed experiment and verify each result CSV."""
    manifest_path = results_dir / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing random experiment manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completed = manifest.get("completed_seeds", {})
    expected_keys = {str(seed) for seed in EXPECTED_SEEDS}
    if (
        manifest.get("all_ten_complete") is not True
        or set(completed) != expected_keys
        or len(set(completed.values())) != len(EXPECTED_SEEDS)
    ):
        raise ValueError(f"Random experiment is not a complete ten-seed run: {manifest_path}")

    result_files = {dataset: [] for dataset in datasets}
    manifest_scores = {dataset: [] for dataset in datasets}
    for seed in EXPECTED_SEEDS:
        seed_dir = results_dir / f"seed_{seed:02d}"
        seed_manifest_path = seed_dir / "seed_manifest.json"
        if not seed_manifest_path.is_file():
            raise FileNotFoundError(f"Missing seed manifest: {seed_manifest_path}")
        seed_manifest = json.loads(seed_manifest_path.read_text(encoding="utf-8"))
        if seed_manifest.get("seed") != seed:
            raise ValueError(f"Unexpected seed in {seed_manifest_path}")
        if seed_manifest.get("model_sha256") != completed[str(seed)]:
            raise ValueError(f"Model hash differs between manifests: {seed_manifest_path}")
        for dataset in datasets:
            csv_path = seed_dir / f"{dataset}_gene_result.csv"
            if not csv_path.is_file():
                raise FileNotFoundError(f"Missing random result: {csv_path}")
            expected_hash = (
                seed_manifest.get("metrics", {}).get(dataset, {}).get("csv_sha256")
            )
            if not expected_hash or _sha256_file(csv_path) != expected_hash:
                raise ValueError(f"CSV hash does not match seed manifest: {csv_path}")
            result_files[dataset].append(csv_path)
            manifest_scores[dataset].append(
                seed_manifest.get("metrics", {}).get(dataset, {}).get("balanced_accuracy_top30")
            )
    return manifest, result_files, manifest_scores


def extract_random_mean_std(datasets, result_files, vocab_tokens):
    """Calculate per-dataset sample mean and s.d. from seeds 1 through 10."""
    means = []
    stds = []
    all_run_accs = []
    
    for dataset in datasets:
        accs = []
        for file_path in result_files[dataset]:
            acc = calculate_metrics(file_path, dataset, vocab_tokens, use_top30=True)
            if acc is None or not np.isfinite(acc):
                raise ValueError(f"{dataset}: invalid balanced accuracy in {file_path}")
            accs.append(acc)
        
        if len(accs) < 2:
            raise ValueError(f"{dataset}: need at least two valid random runs; found {len(accs)}")
        means.append(float(np.mean(accs)))
        stds.append(float(np.std(accs, ddof=1)))  # sample s.d., not s.e.m.
        all_run_accs.append(accs)

    return means, stds, all_run_accs

def extract_weight_accuracies(datasets, base_path_pattern, vocab_tokens):
    """提取权重结果的平衡准确率"""
    accuracies = []
    
    for dataset in datasets:
        file_path = base_path_pattern.format(dataset=dataset)
        if os.path.exists(file_path):
            acc = calculate_metrics(file_path, dataset, vocab_tokens, use_top30=True)
            accuracies.append(acc if acc is not None else np.nan)
        else:
            accuracies.append(np.nan)
    
    return accuracies

def plot_accuracy_comparison(
    datasets,
    random_means,
    random_stds,
    random_run_accs,
    weight_accs,
    output_pdf = "top30_balanced_accuracy.pdf"
) -> None:
    """
    绘制平衡准确率对比柱状图
    
    Parameters:
    -----------
    datasets : list
        数据集名称列表
    random_means : list
        随机实验的均值列表
    random_stds : list
        随机实验的标准差列表
    random_run_accs : list of lists
        每个数据集各次随机初始化的平衡准确率
    weight_accs : list
        权重实验的准确率列表
    output_pdf : str
        输出PDF文件名
    """
    
    print("=" * 60)
    print("绘制平衡准确率对比柱状图")
    print("=" * 60)
    
    # 转换为百分比
    random_means_pct = [v * 100.0 for v in random_means]
    random_stds_pct = [v * 100.0 for v in random_stds]
    weight_accs_pct = [v * 100.0 for v in weight_accs]
    
    # Use the same canvas dimensions as weight.py.
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_IN, dpi=600)
    fig.subplots_adjust(bottom=0.24)
    
    # 设置柱状图的位置 - 调整宽度和间距以增加缝隙
    x = np.arange(len(datasets))
    width = 0.25  # 柱子宽度调窄
    gap = 0.08   # 两个柱子之间的额外间隙
    
    # 直接写死颜色
    color_random = "#7AC3DF"  # scCello的颜色 - 对应Random
    color_weight = "#4EA3F1"  # scGPT的颜色 - 对应Weight
    
    # 绘制随机结果柱状图（带误差线）- 向左偏移
    bars1 = plt.bar(x - width/2 - gap/2, random_means_pct, width, 
                    label='Random', 
                    color=color_random, 
                    alpha=0.8,
                    edgecolor='none', 
                    linewidth=0.0)
    
    # 添加误差线
    plt.errorbar(x - width/2 - gap/2, random_means_pct, yerr=random_stds_pct,
                 fmt='none', ecolor='#222222', elinewidth=1.1,
                 capsize=3, capthick=1.1, zorder=5)
    
    # Fixed offsets show each random initialization without changing its value.
    random_x = x - width/2 - gap/2
    for bar_x, run_accs in zip(random_x, random_run_accs):
        offsets = np.linspace(-0.08, 0.08, len(run_accs))
        ax.scatter(bar_x + offsets, np.asarray(run_accs) * 100.0,
                   s=26, color='#173D50', edgecolors='white', linewidths=0.25,
                   alpha=0.95, zorder=6)

    # 绘制权重结果柱状图 - 向右偏移
    bars2 = plt.bar(x + width/2 + gap/2, weight_accs_pct, width,
                    label='Pre-trained', 
                    color=color_weight, 
                    alpha=0.8,
                    edgecolor='none', 
                    linewidth=0.0)
    
    # 设置图表样式
    plt.xlabel('')
    plt.ylabel('Balanced Accuracy (%)', fontsize=16, fontweight='normal')
    plt.title("")
    
    # 设置X轴标签
    plt.xticks(x, datasets, ha='center', fontsize=16)
    
    # 设置Y轴范围
    plt.ylim(0, 105)
    plt.yticks(np.arange(0, 101, 20), fontsize=14)

    # Chance-level reference line for balanced accuracy.
    plt.axhline(
        y=50,
        color="#9E9E9E",
        linestyle="--",
        linewidth=1.2,
        alpha=0.85,
        zorder=0,
    )
    
    # 添加图例
    ax.legend(frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.06), fontsize=16,
              borderaxespad=0)

    # 不显示背景网格线
    plt.grid(False)
    
    # 移除上、右边框
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#000000')
    ax.spines['bottom'].set_color('#000000')
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(axis='x', labelsize=14, length=0, colors='#000000')
    ax.tick_params(axis='y', labelsize=14, length=0, colors='#000000')
    
    # 保存为PDF
    fig.savefig(output_pdf, dpi=600, bbox_inches='tight', format='pdf')
    output_stem = os.path.splitext(output_pdf)[0]
    fig.savefig(f'{output_stem}.svg', bbox_inches='tight', format='svg')
    fig.savefig(f'{output_stem}.png', dpi=600, bbox_inches='tight', format='png')
    fig.savefig(f'{output_stem}.tiff', dpi=600, bbox_inches='tight', format='tiff')
    plt.close(fig)
    
    print(f"\n✅ 图表已保存为: {output_pdf}")
    
    # 打印统计摘要
    print(f"\n📈 统计摘要:")
    print(f"数据集数量: {len(datasets)}")
    print(f"Random 平均平衡准确率: {np.mean(random_means):.4f} ± {np.mean(random_stds):.4f}")
    print(f"Weight 平均平衡准确率: {np.mean(weight_accs):.4f}")
    
    # 详细对比
    print(f"\n📊 详细对比:")
    for i, ds in enumerate(datasets):
        diff = weight_accs[i] - random_means[i]
        if diff > 0:
            diff_str = f"+{diff:.4f}"
            diff_symbol = "↑"
        elif diff < 0:
            diff_str = f"{diff:.4f}"
            diff_symbol = "↓"
        else:
            diff_str = "0.0000"
            diff_symbol = "="
        
        print(f"  {ds}:")
        print(f"    Random: {random_means[i]:.4f} ± {random_stds[i]:.4f}")
        print(f"    Weight: {weight_accs[i]:.4f}")
        print(f"    差值 (Weight-Random): {diff_str} {diff_symbol}")
    
    print("\n" + "=" * 60)
    print("绘图完成！")
    print("=" * 60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-results-dir", type=Path, default=DEFAULT_RANDOM_RESULTS_DIR)
    parser.add_argument("--weight-results-dir", type=Path, default=DEFAULT_WEIGHT_RESULTS_DIR)
    parser.add_argument("--vocab-path", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF)
    args = parser.parse_args()

    # 数据集列表
    datasets = ['hESC', 'hHep', 'mHSC-E', 'mHSC-GM', 'mHSC-L']
    _, result_files, manifest_scores = validate_random_results(args.random_results_dir, datasets)
    validate_matched_conditions(args.random_results_dir, args.weight_results_dir, datasets, result_files)
    vocab_tokens = load_vocab_tokens(args.vocab_path)
    weight_pattern = str(args.weight_results_dir / "{dataset}_gene_result.csv")
    
    print("=" * 60)
    print("处理 top30 结果 - 平衡准确率")
    print("=" * 60)
    print(f"Random results: {args.random_results_dir} (seeds 1-10)")
    print(f"Weight pattern: {weight_pattern}")
    
    # 提取随机结果（均值和标准差）
    print("\n📊 提取随机实验结果...")
    random_means, random_stds, random_run_accs = extract_random_mean_std(
        datasets, result_files, vocab_tokens
    )
    run_counts = [len(scores) for scores in random_run_accs]
    if len(set(run_counts)) != 1:
        raise ValueError(f"Random run counts differ across datasets: {dict(zip(datasets, run_counts))}")
    if run_counts != [len(EXPECTED_SEEDS)] * len(datasets):
        raise ValueError(f"Expected exactly ten random runs per dataset; found {dict(zip(datasets, run_counts))}")
    print(f"Random runs per dataset: n = {run_counts[0]}; error bars: sample s.d. (ddof=1)")
    for dataset in datasets:
        mismatches = [
            seed for seed, old, new in zip(
                EXPECTED_SEEDS, manifest_scores[dataset], random_run_accs[datasets.index(dataset)]
            )
            if old is None or abs(float(old) - new) > 1e-10
        ]
        if mismatches:
            print(
                f"[INFO] {dataset}: recomputed seeds {mismatches}; stored manifest BA "
                "uses a different scoring policy and is not used in this figure."
            )
    
    for ds, mean, std in zip(datasets, random_means, random_stds):
        print(f"  {ds}: {mean:.4f} ± {std:.4f}")
    
    # 提取权重结果
    print("\n📊 提取权重结果...")
    weight_accs = extract_weight_accuracies(datasets, weight_pattern, vocab_tokens)
    
    for ds, acc in zip(datasets, weight_accs):
        print(f"  {ds}: {acc:.4f}")
    
    # 输出文件名
    output_pdf = args.output_pdf
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    
    # 绘制图表
    plot_accuracy_comparison(datasets, random_means, random_stds, random_run_accs,
                             weight_accs, output_pdf)
    output_stem = Path(output_pdf).with_suffix("")
    summary_path = output_stem.with_name(output_stem.name + "_summary.csv")
    summary_rows = []
    for index, dataset in enumerate(datasets):
        row = {
            "dataset": dataset,
            "random_mean_balanced_accuracy": random_means[index],
            "random_sample_sd": random_stds[index],
            "pretrained_balanced_accuracy": weight_accs[index],
            "pretrained_minus_random": weight_accs[index] - random_means[index],
        }
        row.update({
            f"seed_{seed:02d}_ba": random_run_accs[index][seed - 1]
            for seed in EXPECTED_SEEDS
        })
        row.update({
            f"manifest_seed_{seed:02d}_ba": manifest_scores[dataset][seed - 1]
            for seed in EXPECTED_SEEDS
        })
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, float_format="%.10g")
    print(f"Summary table saved to: {summary_path}")
    caption = (
        f"Fig. 6a | Balanced accuracy for randomly initialized and pretrained scGPT. "
        f"Random bars show the mean across n = {run_counts[0]} runs of randomly "
        "initialized scGPT per dataset; overlaid points show individual runs and error "
        "bars indicate sample standard deviation (s.d.). Pretrained bars show "
        "values from a single deterministic run (n = 1), without error bars. "
        "Balanced accuracy was recalculated on the top 30% of vocabulary-mapped genes "
        "ranked by absolute true change using the evaluator's EPS_DIR=0.001 policy; "
        "near-zero true directions were excluded and near-zero predictions counted as incorrect."
        " Both conditions used the same native per-cell quantile-binned inputs, "
        "fixed dataset-specific preprocessing seeds, no log1p, and 16 EMA updates "
        "with 0.9 retention of the previous state."
    )
    caption_path = str(output_stem) + '_caption.txt'
    with open(caption_path, 'w', encoding='utf-8') as handle:
        handle.write(caption + '\n')
    print(f"Figure caption saved to: {caption_path}")

if __name__ == "__main__":
    main()
