# `plots/fig06a_pretrained_vs_random_scgpt.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

绘制 Random 和 Weight 两个模型在 top30 数据集上的平衡准确率对比柱状图

## Dependencies

Imports found in the source (standard library and external modules): `glob`, `matplotlib`, `numpy`, `os`, `pandas`, `warnings`.

## Defined interfaces

`calculate_metrics`, `extract_random_mean_std`, `extract_weight_accuracies`, `plot_accuracy_comparison`, `main`

## Invocation

Run `python plots/fig06a_pretrained_vs_random_scgpt.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `.png`
- `/mnt/10T/yzn/FoundBench/FBplot/fig4/results_multidataset_pseudotime_227_random/{dataset}_gene_result_run*.csv`
- `/mnt/10T/yzn/benchmark_GRN/pre_scgpt/results_multidataset_pseudotime_227/{dataset}_gene_result.csv`
- `/mnt/10T/yzn/scGRN-Bench/FBplot/fig4/results_multidataset_pseudotime_227_random/{dataset}_gene_result_run*.csv`
- `top30_balanced_accuracy.pdf`
- `top30_balanced_accuracy_no_mDC.pdf`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [fig06a_pretrained_vs_random_scgpt.py](fig06a_pretrained_vs_random_scgpt.py)
