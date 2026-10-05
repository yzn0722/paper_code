# `plots/fig06a_pretrained_vs_random_scgpt.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

绘制 Random 和 Weight 两个模型在 top30 数据集上的平衡准确率对比柱状图

Metric handling uses `in_eval` when present, excludes near-zero true directions,
and counts near-zero predictions as incorrect, matching the scGPT evaluator.
Random results are read only from `seed_01` through `seed_10` under the seeded
experiment directory. The script validates `experiment_manifest.json`, per-seed
CSV hashes, and exactly ten files per dataset before plotting. The default PDF
is `top30_balanced_accuracy_seeded10.pdf`; paths can be overridden with
`--random-results-dir`, `--weight-results-dir`, and `--output-pdf`.
The run also writes matching SVG, PNG, TIFF, caption, and a summary CSV with
per-seed values and per-dataset mean, sample s.d., and pretrained comparison.

## Dependencies

Imports found in the source (standard library and external modules): `argparse`, `hashlib`, `json`, `matplotlib`, `numpy`, `os`, `pandas`, `pathlib`, `warnings`.

## Defined interfaces

`calculate_metrics`, `_sha256_file`, `validate_random_results`, `extract_random_mean_std`, `extract_weight_accuracies`, `plot_accuracy_comparison`, `main`

## Invocation

Run `python plots/fig06a_pretrained_vs_random_scgpt.py` after supplying the external CSV inputs. Use `--random-results-dir`, `--weight-results-dir`, or `--output-pdf` to override the defaults.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `/mnt/10T/yzn/scGRN-Bench/FBplot/fig5/results_multidataset_pseudotime_227_random_seeded_20260929/seed_XX/{dataset}_gene_result.csv`
- `/mnt/10T/yzn/benchmark_GRN/pre_scgpt/results_multidataset_pseudotime_227/{dataset}_gene_result.csv`
- `top30_balanced_accuracy_seeded10.pdf` and matching `.svg`, `.png`, `.tiff`, `_caption.txt`, and `_summary.csv`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [fig06a_pretrained_vs_random_scgpt.py](fig06a_pretrained_vs_random_scgpt.py)
