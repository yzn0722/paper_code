import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from GENIE3 import GENIE3, get_link_list

# -------------------------- 核心配置项（按需修改） --------------------------
# 输入根目录（包含CHIP/Non_CHIP/STRING子目录）
INPUT_ROOT = Path("/mnt/10T/yzn/benchmark_GRN/input_process")
# GENIE3输出根目录（自动创建，与其他模型输出格式对齐）
OUTPUT_ROOT = Path("/mnt/10T/yzn/benchmark_GRN/evl_omipath/output_att500/GENIE3")
# GENIE3参数
GENIE3_PARAMS = {
    "tree_method": "RF",    # 树模型类型：RF（随机森林）/ET（极端树）
    "ntrees": 50,           # 树的数量（可调整：50/100/200）
    "nthreads": 20          # 并行线程数（根据CPU核心数设置）
}
# 临时文件路径（改为字符串类型，适配GENIE3要求）
TEMP_FILE = "temp_gene_weights.tsv"  # 关键修复：从Path改为字符串

# -------------------------- 批量处理函数 --------------------------
def batch_run_genie3():
    # 创建输出根目录
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    # 遍历所有数据类型目录（CHIP/Non_CHIP/STRING）
    data_types = ["CHIP"]
    total_files = 0
    success_files = 0
    
    for data_type in data_types:
        data_type_dir = INPUT_ROOT / data_type
        if not data_type_dir.exists():
            print(f"⚠️ 数据类型目录不存在：{data_type_dir} → 跳过")
            continue
        
        # 分数据类型匹配不同的表达量文件名
        if data_type == "CHIP":
            expr_file_pattern = "*_chip_matched-ExpressionData.csv"  # CHIP目录的文件名模式
        else:
            expr_file_pattern = "*_processed-ExpressionData.csv"     # 其他目录的文件名模式
        
        expr_files = list(data_type_dir.glob(expr_file_pattern))
        if not expr_files:
            print(f"⚠️ {data_type}目录下无表达量文件 → 跳过")
            continue
        
        print(f"\n📌 开始处理{data_type}目录（共{len(expr_files)}个文件）")
        total_files += len(expr_files)
        
        # 逐个处理表达量文件
        for expr_file in tqdm(expr_files, desc=f"{data_type}进度"):
            try:
                # 1. 解析数据集名称
                file_name = expr_file.stem
                if data_type == "CHIP":
                    dataset = file_name.replace("_chip_matched-ExpressionData", "")  # CHIP的后缀
                else:
                    dataset = file_name.replace("_processed-ExpressionData", "")     # 其他目录的后缀
                
                # 2. 读取表达量数据（移除network文件相关逻辑）
                expr_data = pd.read_csv(expr_file, index_col=0)
                if expr_data.empty:
                    raise ValueError(f"表达量文件为空：{expr_file}")
                
                # 提取基因名 + 转换为GENIE3要求的格式（行=样本，列=基因）
                gene_names = expr_data.index.tolist()
                expr_matrix = expr_data.T.values  # 转置：GENIE3要求行是样本，列是基因
                
                # 3. 运行GENIE3计算调控权重
                print(f"\n🔧 运行GENIE3处理{dataset}-{data_type}...")
                vim = GENIE3(
                    expr_matrix,
                    gene_names=gene_names,
                    tree_method=GENIE3_PARAMS["tree_method"],
                    ntrees=GENIE3_PARAMS["ntrees"],
                    nthreads=GENIE3_PARAMS["nthreads"]
                )
                
                # 4. 生成临时权重文件（关键修复：file_name传字符串）
                get_link_list(
                    vim,
                    gene_names=gene_names,
                    maxcount="all",  # 输出所有基因对（自动过滤自调控）
                    file_name=TEMP_FILE  # 已是字符串，适配GENIE3要求
                )
                
                # 5. 读取临时文件并添加表头（移除Gene1过滤逻辑）
                weight_df = pd.read_csv(
                    TEMP_FILE,
                    sep="\t",
                    header=None,
                    names=["Gene1", "Gene2", "EdgeWeight"]
                )
                
                # 6. 仅保留排序逻辑（移除过滤步骤）
                weight_df["EdgeWeight"] = abs(weight_df["EdgeWeight"])  # 取绝对值
                weight_df = weight_df.sort_values("EdgeWeight", ascending=False)
                
                # 7. 保存完整结果（命名格式：GENIE3_数据集_数据类型_emb.tsv）
                output_file = OUTPUT_ROOT / f"GENIE3_{dataset}.tsv"
                weight_df.to_csv(output_file, sep="\t", index=False)
                success_files += 1
                print(f"✅ 保存成功（保留全部边）：{output_file}")
                
            except Exception as e:
                print(f"\n❌ 处理{expr_file.name}失败：{str(e)}")
                continue
            finally:
                # 清理临时文件（兼容字符串路径）
                if os.path.exists(TEMP_FILE):
                    os.remove(TEMP_FILE)
    
    # 批量处理完成统计
    print(f"\n🎉 批量处理完成！")
    print(f"📊 总文件数：{total_files} | 成功数：{success_files} | 失败数：{total_files - success_files}")
    print(f"📁 输出目录：{OUTPUT_ROOT}")

# -------------------------- 执行批量处理 --------------------------
if __name__ == "__main__":
    batch_run_genie3()