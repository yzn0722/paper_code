# `upstream/baselines/deepsem/run_deepsem_batch.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

批量执行：遍历所有*ExpressionData.csv → 转置 → 运行DeepSEM GRN推断
（修正：Gene1/Gene2使用真实基因名，而非细胞名）

## Dependencies

Imports found in the source (standard library and external modules): `numpy`, `os`, `pandas`, `subprocess`, `sys`, `warnings`.

## Defined interfaces

`transpose_scrna_data`, `run_deepsem_grn`

## Invocation

Run `python upstream/baselines/deepsem/run_deepsem_batch.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `   文件名格式：DeepSEM_数据集.tsv`
- `.tsv`
- `/mnt/10T/yzn/DeepSEM-master/main.py`
- `/mnt/10T/yzn/benchmark_GRN/DeepSEM_results`
- `/mnt/10T/yzn/benchmark_GRN/input_process/CHIP`
- `ExpressionData.csv`
- `_expression_transposed.csv`
- `_prediction.csv`
- `_true.csv`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [run_deepsem_batch.py](run_deepsem_batch.py)
