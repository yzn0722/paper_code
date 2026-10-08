#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量执行：遍历所有*ExpressionData.csv → 转置 → 运行DeepSEM GRN推断
（修正：Gene1/Gene2使用真实基因名，而非细胞名）
"""

import os
import sys
import pandas as pd
import numpy as np
import subprocess
import tempfile
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ======================== 请修改这部分参数（仅需改4行）========================
# 1. 数据所在目录（你的CHIP文件夹路径）
DATA_DIR = "/mnt/10T/yzn/benchmark_GRN/input_process/CHIP"
# 2. DeepSEM的main.py文件路径（改成你实际的DeepSEM目录）
DEEPSEM_MAIN_PATH = "/mnt/10T/yzn/DeepSEM-master/main.py"
# 3. 选择任务类型：non_celltype_GRN / celltype_GRN（保持默认即可）
TASK_TYPE = "non_celltype_GRN"
# 4. 结果保存的新文件夹路径
RESULT_DIR = "/mnt/10T/yzn/benchmark_GRN/DeepSEM_results"
# ================================================================================

# -------------------------- 步骤1：转置数据 --------------------------
def transpose_scrna_data(raw_path, transposed_path):
    """转置scRNA-seq CSV文件：基因在行 → 细胞在行"""
    print("="*60)
    print(f"步骤1：转置数据 → {os.path.basename(raw_path)}")
    print("="*60)
    
    if not os.path.exists(raw_path):
        print(f"❌ 错误：原始文件不存在 → {raw_path}")
        sys.exit(1)
    
    try:
        df = pd.read_csv(raw_path, index_col=0)
        print(f"✅ 成功读取原始文件：{df.shape[0]} 基因 × {df.shape[1]} 细胞")
    except Exception as e:
        print(f"❌ 读取原始文件失败：{e}")
        sys.exit(1)
    
    df_transposed = df.T
    df_transposed.to_csv(transposed_path, index=True)
    print(f"✅ 数据转置完成：{df_transposed.shape[0]} 细胞 × {df_transposed.shape[1]} 基因")
    print(f"✅ 转置文件保存至：{transposed_path}")
    print()
    # ✅ 关键修正：返回原始数据的行索引（基因名），而不是转置后的列名
    return transposed_path, df.index.tolist()  # 这里是基因列表！

# Official DeepSEM --setting test writes regulator/target names in this file.
# Preserve its direction (TF -> Target); its internal adjacency is receiver-major.
def read_deepsem_network(result_path, gene_list):
    result_path = Path(result_path)
    if not result_path.is_file():
        raise FileNotFoundError(f"DeepSEM did not write its trained network: {result_path}")
    frame = pd.read_csv(result_path, sep="\t", dtype=str, keep_default_na=False)
    if {"TF", "Target", "EdgeWeight"}.issubset(frame.columns):
        frame = frame.rename(columns={"TF": "Gene1", "Target": "Gene2"})
    required = ["Gene1", "Gene2", "EdgeWeight"]
    if not set(required).issubset(frame.columns):
        raise ValueError(f"Unexpected DeepSEM network columns: {frame.columns.tolist()}")
    frame = frame[required].copy()
    genes = [str(g) for g in gene_list]
    if len(genes) != len(set(genes)):
        raise ValueError("Expression input contains duplicate gene names")
    unknown = (set(frame.Gene1) | set(frame.Gene2)) - set(genes)
    if unknown:
        raise ValueError(f"DeepSEM output contains unknown gene names: {sorted(unknown)[:10]}")
    frame["EdgeWeight"] = pd.to_numeric(frame["EdgeWeight"], errors="raise")
    if not np.isfinite(frame["EdgeWeight"].to_numpy(dtype=float)).all():
        raise ValueError("DeepSEM output contains non-finite weights")
    if frame.duplicated(["Gene1", "Gene2"]).any():
        raise ValueError("DeepSEM output contains duplicate directed edges")
    frame = frame.loc[(frame.Gene1 != frame.Gene2) & (frame.EdgeWeight != 0)]
    if frame.empty:
        raise ValueError("DeepSEM produced no nonzero non-self edges")
    # Keep all trained nonzero scores, including small/negative weights, unchanged.
    return frame.sort_values("EdgeWeight", key=abs, ascending=False, kind="stable").reset_index(drop=True)


# -------------------------- 步骤2：运行DeepSEM并导出真实网络 --------------------------
def run_deepsem_grn(transposed_data_path, save_name, task_type, gene_list, dataset_name):
    """运行DeepSEM并生成Gene1/Gene2/EdgeWeight格式TSV（Gene1/Gene2为真实基因名）"""
    print("="*60)
    print(f"步骤2：运行DeepSEM → {os.path.basename(save_name)}")
    print("="*60)
    
    if not os.path.exists(DEEPSEM_MAIN_PATH):
        print(f"❌ 错误：DeepSEM的main.py不存在 → {DEEPSEM_MAIN_PATH}")
        sys.exit(1)
    
    # 根据任务类型设置最优超参数
    alpha, beta, n_epochs = 100, 1, 120
    if task_type == "celltype_GRN":
        alpha = 0.1 if dataset_name in ["hESC", "mHSC-E"] else 1
        beta, n_epochs = 0.01, 150

    # A fresh run directory prevents an old network from masquerading as new output.
    save_prefix = Path(save_name).resolve()
    save_prefix.parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=save_prefix.name + "_", dir=save_prefix.parent))
    deepsem_cmd = [
        sys.executable, str(Path(DEEPSEM_MAIN_PATH).resolve()),
        "--task", task_type, "--data_file", str(Path(transposed_data_path).resolve()),
        "--save_name", str(run_dir), "--setting", "test",
        "--alpha", str(alpha), "--beta", str(beta), "--n_epochs", str(n_epochs),
    ]
    
    print("📝 运行命令：")
    print(subprocess.list2cmdline(deepsem_cmd))
    print()
    print("🚀 开始训练DeepSEM（实时输出Epoch日志）...")
    print("-"*60)
    
    # Preserve the actual model output and training log for diagnosis.
    with (run_dir / "training.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            deepsem_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            cwd=run_dir,
        )
        
        # 实时打印日志并保存
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.strip())
                log.write(line)
        
        return_code = process.poll()
        
        if return_code != 0:
            raise RuntimeError(f"DeepSEM failed with exit code {return_code}; see {run_dir / 'training.log'}")

    frame = read_deepsem_network(run_dir / "GRN_inference_result.tsv", gene_list)
    destination = Path(RESULT_DIR) / f"DeepSEM_{dataset_name}.tsv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, sep="\t", index=False, encoding="utf-8")
    print(f"✅ 已导出真实 DeepSEM 网络：{destination} ({len(frame)} edges)")
    return destination

# -------------------------- 批量主流程 --------------------------
if __name__ == "__main__":
    expr_files = [f for f in os.listdir(DATA_DIR) if f.endswith("ExpressionData.csv")]
    if not expr_files:
        print("❌ 未找到任何*ExpressionData.csv文件！")
        sys.exit(1)
    
    print(f"📂 发现 {len(expr_files)} 个数据集：")
    for f in expr_files:
        print(f"  - {f}")
    print()
    
    for expr_file in expr_files:
        dataset_name = expr_file.split("_")[0]
        print(f"\n{'='*70}")
        print(f"📌 开始处理数据集：{dataset_name}")
        print(f"{'='*70}")
        
        raw_csv_path = os.path.join(DATA_DIR, expr_file)
        transposed_csv_path = os.path.join(DATA_DIR, f"{dataset_name}_expression_transposed.csv")
        temp_save_name = os.path.join(DATA_DIR, f"temp_DeepSEM_{dataset_name}")
        
        # 步骤1：转置数据（返回基因列表）
        transposed_path, gene_list = transpose_scrna_data(raw_csv_path, transposed_csv_path)
        print(f"✅ 检测到 {len(gene_list)} 个基因：{gene_list[:5]}...")  # 打印前5个基因验证
        
        # 步骤2：运行DeepSEM并生成正确TSV
        run_deepsem_grn(transposed_path, temp_save_name, TASK_TYPE, gene_list, dataset_name)
        
        print(f"\n✅ 数据集 {dataset_name} 处理完成！")
        print()
    
    print("\n" + "="*70)
    print("🎉 所有数据集处理完成！")
    print("="*70)
    print(f"💡 所有GRN结果保存在：{RESULT_DIR}")
    print("   文件名格式：DeepSEM_数据集.tsv")
    print("   列名格式：Gene1(基因) | Gene2(基因) | EdgeWeight")
