
import pandas as pd
import numpy as np
import os
from sklearn.metrics import average_precision_score
# GRNboost 核心库
try:
    from arboreto.algo import grnboost2
except ModuleNotFoundError:
    raise ModuleNotFoundError("请先安装arboreto：pip install arboreto -i https://pypi.tuna.tsinghua.edu.cn/simple")

# ===================== 1. 核心配置（已适配你的路径） =====================
# 数据集列表（可扩展）
  # 先处理hESC，如需批量添加hHep/mDC等直接加列表
# 基础路径模板（替换为你的实际路径）
DATASETS = ["hESC", "hHep", "mDC", "mHSC-E", "mHSC-L", "mHSC-GM"]
BASE_PATH = "/mnt/10T/yzn/benchmark_GRN/input_process/STRING"
#EXPR_TEMPLATE = f"{BASE_PATH}/{{dataset}}_chip_matched-ExpressionData.csv"  # 表达矩阵
EXPR_TEMPLATE = f"{BASE_PATH}/{{dataset}}_processed-ExpressionData.csv"  # 表达矩阵
#NETWORK_TEMPLATE = f"{BASE_PATH}/{{dataset}}_chip_matched-network.csv"      # 真实网络
NETWORK_TEMPLATE = f"{BASE_PATH}/{{dataset}}_processed-network.csv"      # 真实网络
# 输出目录（自动创建）
#OUT = "/mnt/10T/yzn/benchmark_GRN/evl_omipath/output_emb500"
OUTPUT_DIR = f"{BASE_PATH}/GRNboost"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== 2. 自动提取TF列表（从真实网络的Gene1列） =====================
def extract_tf_list(dataset):
    """从真实网络中提取TF列表（Gene1列），生成TF_FILE"""
    network_file = NETWORK_TEMPLATE.format(dataset=dataset)
    tf_file = f"{OUTPUT_DIR}/{dataset}_tf_list.txt"
    
    # 读取真实网络
    network_df = pd.read_csv(network_file)
    # 数据清洗：去重、去自环边
    network_df = network_df[['Gene1', 'Gene2']].copy()
    network_df['Gene1'] = network_df['Gene1'].astype(str).str.strip()
    network_df['Gene2'] = network_df['Gene2'].astype(str).str.strip()
    network_df = network_df[network_df['Gene1'] != network_df['Gene2']]
    network_df.drop_duplicates(keep='first', inplace=True)
    
    # 提取TF列表（Gene1列）
    tf_list = sorted(list(set(network_df['Gene1'])))
    # 保存TF_FILE
    with open(tf_file, 'w', encoding='utf-8') as f:
        for tf in tf_list:
            f.write(tf + '\n')
    
    print(f"✅ {dataset} - TF列表生成完成：{tf_file}，共{len(tf_list)}个TF")
    return tf_file, tf_list

# ===================== 3. 加载并预处理表达矩阵（核心：强制转置） =====================
def load_expression_data(dataset):
    """加载表达矩阵并预处理（转置+去空值+标准化）"""
    expr_file = EXPR_TEMPLATE.format(dataset=dataset)
    # 读取表达矩阵（原始：行=基因，列=样本）
    expr_df = pd.read_csv(expr_file, index_col=0)  # 假设第一列是基因名
    
    # ========== 核心修正：强制转置（基因×样本 → 样本×基因） ==========
    expr_df = expr_df.T  # 转置！转置！转置！
    print(f"🔄 {dataset} - 表达矩阵已转置：原始维度{expr_df.T.shape} → 转置后{expr_df.shape}")
    
    # 数据清洗
    expr_df = expr_df.dropna(axis=1)  # 删除有缺失值的基因（转置后列是基因）
    expr_df = expr_df.loc[:, (expr_df != 0).any(axis=0)]  # 删除全0基因
    # 表达量标准化（均值0，方差1）
    expr_df = (expr_df - expr_df.mean(axis=0)) / expr_df.std(axis=0)
    
    print(f"✅ {dataset} - 表达矩阵加载完成：维度{expr_df.shape}（样本数×基因数）")
    return expr_df

# ===================== 4. GRNboost 核心推断（最终稳定版） =====================
def run_grnboost_inference(dataset, expr_df, tf_list):
    """运行GRNboost2推断GRN（适配所有版本，移除n_jobs/client参数）"""
    output_file = f"{OUTPUT_DIR}/{dataset}_grnboost_result.tsv"
    
    # 过滤TF列表（仅保留表达矩阵中存在的TF）
    valid_tf_list = [tf for tf in tf_list if tf in expr_df.columns]
    if len(valid_tf_list) == 0:
        raise ValueError(f"{dataset} - 无有效TF（TF列表与表达矩阵无交集）")
    print(f"✅ {dataset} - 有效TF数量：{len(valid_tf_list)}（过滤后）")
    
    # 核心：GRNboost2推断（移除所有不兼容参数，适配所有版本）
    print(f"📌 {dataset} - 开始GRNboost推断（耗时较长，请耐心等待）...")
    grn_result = grnboost2(
        expression_data=expr_df,
        tf_names=valid_tf_list,
        seed=42,  # 固定种子保证可复现
        verbose=True  # 打印进度
    )
    
    # 结果处理：重命名列、去自环边、排序
    grn_result = grn_result.rename(columns={
        'TF': 'Gene1',
        'target': 'Gene2',
        'importance': 'EdgeWeight'
    })
    grn_result = grn_result[grn_result['Gene1'] != grn_result['Gene2']]  # 去自环边
    grn_result = grn_result.sort_values('EdgeWeight', ascending=False)  # 按权重降序
    grn_result = grn_result.drop_duplicates(subset=['Gene1', 'Gene2'], keep='first')  # 去重
    
    # 保存结果
    grn_result.to_csv(output_file, sep='\t', index=False)
    print(f"✅ {dataset} - GRN推断完成，结果保存至：{output_file}")
    print(f"✅ {dataset} - 推断出{len(grn_result)}条调控边")
    
    return output_file, grn_result

# ===================== 5. AUPR Ratio 计算（和参考代码完全对齐） =====================
def calculate_aupr_ratio(dataset, grn_file, label_df):
    """计算AUPR Ratio"""
    # 读取GRNboost推断结果
    grn_df = pd.read_csv(grn_file, sep='\t')
    grn_df['EdgeWeight'] = abs(grn_df['EdgeWeight'])  # 权重取绝对值
    grn_df = grn_df.sort_values('EdgeWeight', ascending=False)
    
    # 提取TF和基因集合
    TFs = set(label_df['Gene1'])
    Genes = set(label_df['Gene1']) | set(label_df['Gene2'])
    
    # TF过滤
    grn_df = grn_df[grn_df['Gene1'].apply(lambda x: x in TFs)]
    grn_df = grn_df[grn_df['Gene2'].apply(lambda x: x in Genes)]
    
    # 构建真实边集合（分隔符避免冲突）
    label_set = set(label_df['Gene1'] + "|" + label_df['Gene2'])
    
    # 构建预测权重字典
    res_d = {}
    for _, row in grn_df.iterrows():
        edge_key = row['Gene1'] + "|" + row['Gene2']
        res_d[edge_key] = row['EdgeWeight']
    
    # 遍历所有TF×Gene组合
    l = []  # 真实标签
    p = []  # 预测权重
    for tf in TFs:
        for gene in Genes:
            if tf == gene:
                continue
            edge_key = tf + "|" + gene
            # 真实标签
            l.append(1 if edge_key in label_set else 0)
            # 预测权重（无则赋值-1）
            p.append(res_d.get(edge_key, -1))
    
    # 计算AUPR和AUPR Ratio
    l = np.array(l)
    p = np.array(p)
    aupr = average_precision_score(l, p)
    random_baseline = np.mean(l)
    aupr_ratio = aupr / random_baseline if random_baseline > 0 else 0.0
    
    # 输出结果
    print(f"\n📊 {dataset} - AUPR评估结果：")
    print(f"   真实正样本数：{sum(l)}")
    print(f"   总边数：{len(l)}")
    print(f"   随机基线：{random_baseline:.6f}")
    print(f"   AUPR：{aupr:.6f}")
    print(f"   AUPR Ratio：{aupr_ratio:.6f}")
    
    return {
        'AUPR': round(aupr, 6),
        'Random_Baseline': round(random_baseline, 6),
        'AUPR_Ratio': round(aupr_ratio, 6)
    }

# ===================== 6. EPR 计算（和参考代码完全对齐） =====================
def calculate_epr(dataset, grn_file, label_df):
    """计算EPR（Exact Precision Ratio）"""
    # 读取GRNboost推断结果
    grn_df = pd.read_csv(grn_file, sep='\t')
    grn_df['EdgeWeight'] = abs(grn_df['EdgeWeight'])  # 权重取绝对值
    grn_df = grn_df.sort_values('EdgeWeight', ascending=False)
    
    # 提取TF和基因集合
    TFs = set(label_df['Gene1'])
    Genes = set(label_df['Gene1']) | set(label_df['Gene2'])
    
    # TF过滤
    grn_df = grn_df[grn_df['Gene1'].apply(lambda x: x in TFs)]
    grn_df = grn_df[grn_df['Gene2'].apply(lambda x: x in Genes)]
    
    # 构建真实边集合
    label_set = set(label_df['Gene1'] + "|" + label_df['Gene2'])
    true_edges_count = len(label_set)  # 真实边总数（K）
    
    # 取Top-K条预测边（K=真实边总数）
    top_k = true_edges_count
    if len(grn_df) < top_k:
        top_k = len(grn_df)
    top_k_edges = grn_df.head(top_k)
    
    # 计算Top-K中命中的真实边数
    top_k_edge_set = set(top_k_edges['Gene1'] + "|" + top_k_edges['Gene2'])
    correct_count = len(top_k_edge_set & label_set)
    
    # 计算随机基线正确数
    total_possible_edges = len(TFs) * len(Genes) - len(TFs)  # 总边数-自环边
    random_correct = (true_edges_count **2) / total_possible_edges if total_possible_edges > 0 else 0.0
    
    # 计算EPR
    epr = correct_count / random_correct if random_correct > 0 else 0.0
    
    # 输出结果
    print(f"\n📊 {dataset} - EPR评估结果：")
    print(f"   真实边总数（K）：{true_edges_count}")
    print(f"   Top-K预测边数：{top_k}")
    print(f"   Top-K命中真实边数：{correct_count}")
    print(f"   随机基线正确数：{random_correct:.6f}")
    print(f"   EPR：{epr:.6f}")
    
    return {
        'True_Edges_Count': true_edges_count,
        'Top_K_Correct': correct_count,
        'Random_Baseline_Correct': round(random_correct, 6),
        'EPR': round(epr, 6)
    }

# ===================== 7. 主函数：一键运行（GRN推断 + AUPR + EPR） =====================
def main():
    all_eval_results = []
    for dataset in DATASETS:
        print(f"\n{'='*80}")
        print(f"📌 开始处理数据集：{dataset}")
        print(f"{'='*80}")
        
        # 步骤1：读取真实网络（复用，避免重复读取）
        network_file = NETWORK_TEMPLATE.format(dataset=dataset)
        label_df = pd.read_csv(network_file)
        label_df = label_df[['Gene1', 'Gene2']].copy()
        label_df['Gene1'] = label_df['Gene1'].astype(str).str.strip()
        label_df['Gene2'] = label_df['Gene2'].astype(str).str.strip()
        label_df = label_df[label_df['Gene1'] != label_df['Gene2']]
        label_df.drop_duplicates(keep='first', inplace=True)
        
        # 步骤2：提取TF列表
        tf_file, tf_list = extract_tf_list(dataset)
        
        # 步骤3：加载表达矩阵（自动转置）
        expr_df = load_expression_data(dataset)
        
        # 步骤4：GRNboost推断
        grn_file, _ = run_grnboost_inference(dataset, expr_df, tf_list)
        
        # 步骤5：计算AUPR Ratio
        aupr_result = calculate_aupr_ratio(dataset, grn_file, label_df)
        
        # 步骤6：计算EPR
        epr_result = calculate_epr(dataset, grn_file, label_df)
        
        # 合并所有结果
        final_result = {
            'Dataset': dataset,
            **aupr_result,
            **epr_result
        }
        all_eval_results.append(final_result)
    
    # 保存评估汇总结果（包含AUPR+EPR）
    eval_df = pd.DataFrame(all_eval_results)
    # 调整列顺序（便于查看）
    columns_order = [
        'Dataset', 'True_Edges_Count', 'AUPR', 'AUPR_Ratio', 
        'Random_Baseline', 'Top_K_Correct', 'Random_Baseline_Correct', 'EPR'
    ]
    eval_df = eval_df[columns_order]
    eval_df.to_csv(f"{OUTPUT_DIR}/grnboost_STR_summary.csv", index=False)
    
    # 打印最终汇总
    print(f"\n{'='*80}")
    print(f"🎉 所有流程完成！汇总结果保存至：{OUTPUT_DIR}/grnboost_aupr_epr_summary.csv")
    print(f"{'='*80}")
    print("\n📊 最终评估汇总（AUPR + EPR）：")
    print(eval_df.to_string(index=False))

if __name__ == "__main__":
    main()