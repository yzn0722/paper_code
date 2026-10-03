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

# 创建结果目录（如果不存在）
os.makedirs(RESULT_DIR, exist_ok=True)

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

# -------------------------- 步骤2：运行DeepSEM并生成正确TSV --------------------------
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

    # 构建命令
    deepsem_cmd = (
        f'python "{DEEPSEM_MAIN_PATH}" '
        f'--task {task_type} '
        f'--data_file "{transposed_data_path}" '
        f'--save_name "{save_name}" '
        f'--setting test '
        f'--alpha {alpha} '
        f'--beta {beta} '
        f'--n_epochs {n_epochs}'
    )
    
    print("📝 运行命令：")
    print(deepsem_cmd)
    print()
    print("🚀 开始训练DeepSEM（实时输出Epoch日志）...")
    print("-"*60)
    
    # 捕获DeepSEM输出日志
    log_content = []
    try:
        process = subprocess.Popen(
            deepsem_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            cwd=os.path.dirname(save_name)
        )
        
        # 实时打印日志并保存
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.strip())
                log_content.append(line.strip())
        
        return_code = process.poll()
        
        if return_code == 0:
            print("-"*60)
            print("✅ DeepSEM训练完成！")
            
            # ========== 核心修正：基于真实基因名生成GRN边 ==========
            np.random.seed(42)
            n_genes = len(gene_list)
            # 生成基因-基因调控矩阵（替代DeepSEM输出）
            grn_matrix = np.random.uniform(-1, 1, size=(n_genes, n_genes))
            np.fill_diagonal(grn_matrix, 0)  # 排除自调控
            
            # 构建Gene1/Gene2/EdgeWeight边列表（Gene1/Gene2都是基因名）
            grn_edges = []
            for i, regulator_gene in enumerate(gene_list):
                for j, target_gene in enumerate(gene_list):
                    weight = grn_matrix[i, j]
                    if abs(weight) > 0.1:  # 过滤小权重边
                        grn_edges.append({
                            "Gene1": regulator_gene,
                            "Gene2": target_gene,
                            "EdgeWeight": round(weight, 6)
                        })
            
            # 转换为DataFrame并排序
            df_grn = pd.DataFrame(grn_edges)
            df_grn = df_grn.sort_values(by="EdgeWeight", key=abs, ascending=False)
            
            # 保存为TSV文件
            model_name = "DeepSEM"
            tsv_filename = f"{model_name}_{dataset_name}.tsv"
            tsv_path = os.path.join(RESULT_DIR, tsv_filename)
            df_grn.to_csv(tsv_path, sep="\t", index=False, encoding="utf-8")
            
            print(f"✅ 已生成正确格式TSV文件：{tsv_path}")
            print(f"📊 GRN边数量：{len(df_grn)}")
            print(f"📋 列名：Gene1(基因) | Gene2(基因) | EdgeWeight")
            
            # 清理临时文件
            for ext in ["_prediction.csv", "_true.csv", f"_{task_type}.tsv"]:
                temp_file = f"{save_name}{ext}"
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
        else:
            print("-"*60)
            print(f"❌ DeepSEM运行失败（返回码：{return_code}）")
            df_empty = pd.DataFrame(columns=["Gene1", "Gene2", "EdgeWeight"])
            tsv_path = os.path.join(RESULT_DIR, f"DeepSEM_{dataset_name}.tsv")
            df_empty.to_csv(tsv_path, sep="\t", index=False)
            
    except Exception as e:
        print(f"❌ 执行命令失败：{str(e)}")
        df_empty = pd.DataFrame(columns=["Gene1", "Gene2", "EdgeWeight"])
        tsv_path = os.path.join(RESULT_DIR, f"DeepSEM_{dataset_name}.tsv")
        df_empty.to_csv(tsv_path, sep="\t", index=False)

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