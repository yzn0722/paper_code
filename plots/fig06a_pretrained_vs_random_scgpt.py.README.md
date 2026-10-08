# `plots/fig06a_pretrained_vs_random_scgpt.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

绘制 Random 和 Weight 两个模型在 top30 数据集上的平衡准确率对比柱状图

Metric handling selects the top 30% of vocabulary-mapped genes ranked by
absolute `delta_true` (or uses the exact `in_eval` mask when present), then
recomputes directions from `delta_true` and `delta_pred` with `EPS_DIR=0.001`.
Near-zero true directions are excluded and near-zero predictions count as
incorrect, matching the scGPT evaluator. Legacy result CSVs without mapping
columns require the supplied scGPT vocabulary; gene symbols are uppercased for
membership checks. Stored manifest BA values are retained in the summary for
comparison but are not used when they disagree with the recalculated metric.
Random results are read only from `seed_01` through `seed_10` under the seeded
experiment directory. The script validates `experiment_manifest.json`, per-seed
CSV hashes, and exactly ten files per dataset before plotting. The default PDF
is `top30_balanced_accuracy_seeded10_mapped_top30.pdf`; paths can be overridden
with `--random-results-dir`, `--weight-results-dir`, `--vocab-path`, and
`--output-pdf`.
The run also writes matching SVG, PNG, TIFF, caption, and a summary CSV with
per-seed values and per-dataset mean, sample s.d., and pretrained comparison.

Both conditions must have an identical recorded native-binning/EMA-0.9 protocol.
The script verifies exact equality of observed early/late means, delta_true,
gene mapping and in_eval across all 50 random CSVs and the pretrained condition.
It rejects legacy predictions even if their scores can be recalculated.

## Dependencies

Imports found in the source (standard library and external modules): `argparse`, `hashlib`, `json`, `matplotlib`, `numpy`, `os`, `pandas`, `pathlib`, `warnings`.

## Defined interfaces

`calculate_metrics`, `_sha256_file`, `validate_random_results`, `extract_random_mean_std`, `extract_weight_accuracies`, `plot_accuracy_comparison`, `main`

## Invocation

Run `python plots/fig06a_pretrained_vs_random_scgpt.py` after supplying the external CSV inputs and scGPT vocabulary. Use `--random-results-dir`, `--weight-results-dir`, `--vocab-path`, or `--output-pdf` to override the defaults.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `/mnt/10T/yzn/paper-code/outputs/fig06a_native_binning_ema09_20261008/random/seed_XX/{dataset}_gene_result.csv`
- `/mnt/10T/yzn/paper-code/outputs/fig06a_native_binning_ema09_20261008/pretrained/{dataset}_gene_result.csv`
- `top30_balanced_accuracy_seeded10_mapped_top30.pdf` and matching `.svg`, `.png`, `.tiff`, `_caption.txt`, and `_summary.csv`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. This script reevaluates existing predictions; it does not train or rerun random initializations.

Source: [fig06a_pretrained_vs_random_scgpt.py](fig06a_pretrained_vs_random_scgpt.py)
