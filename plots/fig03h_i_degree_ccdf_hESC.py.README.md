# `plots/fig03h_i_degree_ccdf_hESC.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

基因调控网络度分布可视化 - Nature风格 CCDF版 (模型 vs STRING真实网络)
说明:
- 使用互补累积分布函数 (CCDF) 替代简单的频率分布，以减少尾部噪声并更清晰展示幂律特征
- 叠加 STRING 真实网络（*_processed-network.csv）的入/出度 CCDF 曲线用于对比
- 关键修改：
  1. 模型文件仅保留真实网络中存在的Gene1
  2. 模型文件仅保留真实网络中Gene1/Gene2并集中的Gene2
  3. 模型文件选取的边数与对应真实网络边数一致

## Dependencies

Imports found in the source (standard library and external modules): `argparse`, `fig2_palette`, `matplotlib`, `numpy`, `pandas`, `pathlib`, `warnings`.

## Defined interfaces

`detect_weight_column`, `load_model_data`, `load_real_data`, `get_ccdf`, `format_label_with_tail_prob`, `_to_undirected_degree_series`, `plot_ccdf_distribution`, `main`

## Command-line parameters

- `--extraction`: Extraction subdir under evl_omipath (e.g., emb500, att500, embhidden500).
- `--undirected`: Treat edges as undirected and plot a single degree CCDF.

## Invocation

Run `python plots/fig03h_i_degree_ccdf_hESC.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `.tsv`
- `/mnt/10T/yzn/benchmark_GRN/evl_omipath`
- `/mnt/10T/yzn/benchmark_GRN/input_process/STRING`
- `_CCDF_degree_undirected_models_vs_STRING.pdf`
- `_CCDF_indegree_models_vs_STRING.pdf`
- `_CCDF_outdegree_models_vs_STRING.pdf`
- `_processed-network.csv`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [fig03h_i_degree_ccdf_hESC.py](fig03h_i_degree_ccdf_hESC.py)
