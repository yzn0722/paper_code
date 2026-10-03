# `upstream/dynamics/run_scprint_pseudotime.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

scPRINT 伪时间方向基准（与 `run_unified_multidataset_pseudotime.py` 任务对齐的独立入口）

## scPRINT 如何做「类似」任务（与 scGPT / Geneformer 的差异）

- **Geneformer / LangCell**：离散基因 token + MLM / swap，在序列空间迭代改写 token，用 rank 或隐式表达变化。
- **scGPT**：连续 binned expression + 基因顺序固定，每步用 `mlm_output` EMA 更新，再与 early 平均求 `pred_delta`。
- **scPRINT**（见 `scprint/tasks/denoise.py`、`scprint/model/model.py::_predict`）：
  - 输入是 **连续表达** + `scdataloader.Collator` 选出的基因子集（最多 `max_len`）。
  - 推理主路径是 **`predict_mode="denoise"`**：`forward(..., depth_mult=expression.sum(1)*depth_mult, req_depth=depth*depth_mult)`，
    得到 ZINB 头的 **`mean`**（及可选 disp / zero），这是 **去噪/重构式** 预测，不是按伪时间「生成晚期状态」的显式目标。
  - 官方 `Denoiser` 走完整 DataLoader，并把结果写入 `h5ad`；本脚本与 **`my_GRN_code/infer_raw_attn_to_tsv.py`** 一致：
    **`torch.load` + `inspect.signature` 过滤 `hyper_parameters`**（避免 ckpt 里 `memmap_gene_emb` 等字段触发 Lightning/jsonargparse 报错）、
    可选 **`--token-pkl` 对齐 gene embedding**、**`disable_bias_runtime` forward patch**；
    迭代阶段使用与 **`infer_grn_fp32_noamp_slice_attn`** 相同的 **`gene_pos` 张量 + 按 batch `_predict(denoise)`**（默认 **fp32 无 AMP**），对早期细胞做 EMA 更新后再算方向准确率。

依赖（与官方 README 一致）：`scprint`、`scdataloader`、`lamin`/`lamindb`、`scanpy`、`anndata` 等；首次运行通常需要
`scdataloader` 的 ontology：`populate_my_ontology(...)`。

用法示例：

  python run_scprint_multidataset_pseudotime.py \
    --checkpoint /path/to/model.ckpt \
    --scprint-repo /mnt/10T/yzn/scPRINT \
    --outdir pre_scprint_results \
    --save-trajectory-plot --trajectory-embed umap

## Dependencies

Imports found in the source (standard library and external modules): `__future__`, `anndata`, `argparse`, `inspect`, `json`, `matplotlib`, `mygene`, `numpy`, `os`, `pandas`, `pathlib`, `pickle`, `re`, `scanpy`, `scdataloader`, `scipy`, `scprint`, `sys`, `torch`, `types`, `typing`, `umap`.

## Defined interfaces

`read_pt_file`, `organism_id`, `direction_accuracy_top_genes`, `_pca_2d`, `save_trajectory_plot`, `_fix_genes_in_ckpt_dict`, `_normalize_state_dict_keys`, `_clean_symbol`, `load_env_genes_from_token_pkl`, `_model_genes_mostly_human_ensembl`, `_mygene_taxon_for_symbol_map`, `map_symbols_to_ensembl`, `disable_bias_runtime`, `load_scprint_model`, `expr_to_ensembl_rows`, `pick_eval_genes`, `run_one_dataset`, `parse_args`, `main`

## Command-line parameters

- `--checkpoint`: Lightning ckpt 路径（scPrint）
- `--scprint-repo`: scPRINT 源码根目录（加入 sys.path）
- `--outdir`
- `--datasets`: 逗号分隔子集，默认全部
- `--seed`
- `--save-cell-preds-by-iter`: Save per-iteration per-early-cell denoise+EMA state (disk-heavy).
- `--pt-quantile`
- `--top-percent`
- `--gen-iters`
- `--batch-size`
- `--max-len`: 评估面板最大基因数（与 infer / Denoiser 类似）
- `--predict-depth-mult`: 对应 Denoiser.predict_depth_mult
- `--ema-alpha`: EMA retention on previous state (paper α): x <- alpha*x + (1-alpha)*x_hat.
- `--normalize-target-sum`: scanpy normalize_total；<=0 跳过
- `--log1p`
- `--no-log1p`
- `--transformer`: 加载 ckpt 时传入 scPrint；无 Flash 时用 normal
- `--token-pkl`: Geneformer 式 token_dictionary.pkl：与 infer_raw_attn_to_tsv 一致，按 pkl 基因顺序对齐 embedding；并提供 symbol->Ensembl 评估路径
- `--gene-emb-offset`: 对齐 ckpt embedding 行时的偏移（与 CKPT_GENE_EMB_OFFSET 相同）
- `--no-patch-disable-bias`: 默认会 patch forward 去掉 attn bias（与 GRN 推断脚本一致）；若与某 ckpt 不兼容可加此项
- `--use-amp`: GPU 上使用 autocast（默认 fp32，与 infer_grn_fp32_noamp_slice_attn 一致）
- `--fix-genes-ckpt`: torch.load 修复 hyper_parameters['genes'] 为列表再加载（部分 ckpt 需要）
- `--populate-ontology`: 首次运行建议开启（scdataloader）
- `--no-populate-ontology`
- `--print-every`
- `--acc-eps`: 方向准确率：|true_delta|>eps 才计分（0 与 unified 默认一致）
- `--save-trajectory-plot`
- `--trajectory-embed`

## Invocation

Run `python upstream/dynamics/run_scprint_pseudotime.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `/mnt/10T/yzn/benchmark_GRN/PseudoTime/hESC/PseudoTime.csv`
- `/mnt/10T/yzn/benchmark_GRN/PseudoTime/hHep/PseudoTime.csv`
- `/mnt/10T/yzn/benchmark_GRN/PseudoTime/mDC/PseudoTime.csv`
- `/mnt/10T/yzn/benchmark_GRN/PseudoTime/mHSC-E/PseudoTime.csv`
- `/mnt/10T/yzn/benchmark_GRN/PseudoTime/mHSC-GM/PseudoTime.csv`
- `/mnt/10T/yzn/benchmark_GRN/PseudoTime/mHSC-L/PseudoTime.csv`
- `/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/hESC_chip_matched-ExpressionData.csv`
- `/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/hHep_chip_matched-ExpressionData.csv`
- `/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/mDC_chip_matched-ExpressionData.csv`
- `/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/mHSC-E_chip_matched-ExpressionData.csv`
- `/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/mHSC-GM_chip_matched-ExpressionData.csv`
- `/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/mHSC-L_chip_matched-ExpressionData.csv`
- `/mnt/10T/yzn/scPRINT`
- `2d.csv`
- `2d.png`
- `accuracy_curves.json`
- `diagnostics.json`
- `errors.json`
- `per_gene_final_changes.csv`
- `pred_delta_by_iter.npy`
- `preds_full_by_iter.npy`
- `trajectory_embed_input_matrix.npy`
- `trajectory_raw_matrix.npy`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [run_scprint_pseudotime.py](run_scprint_pseudotime.py)
