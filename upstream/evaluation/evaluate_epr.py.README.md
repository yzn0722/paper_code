# `upstream/evaluation/evaluate_epr.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

GRN early precision ratio evaluation script
Batch EPR evaluation across models, datasets, and ground-truth types.
Author note: optimized from the original implementation.
Version: 1.0

## Dependencies

Imports found in the source (standard library and external modules): `argparse`, `datetime`, `itertools`, `json`, `numpy`, `os`, `pandas`, `sys`, `traceback`, `warnings`.

## Defined interfaces

`EarlyPrec`, `save_results_to_csv`, `evaluate_all_combinations`, `main`

## Command-line parameters

- `--pred_root`: Prediction root path
- `--true_root`: Ground-truth root directory
- `--output`: Output CSV path
- `--tf_edges`: Enable TF filtering (default: True)
- `--config`: JSON config path (overrides CLI arguments)
- `--models`: Model name list
- `--datasets`: Dataset name list
- `--gt_types`: Ground-truth type list

## Invocation

Run `python upstream/evaluation/evaluate_epr.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `.csv`
- `.tsv`
- `_chip_matched-network.csv`
- `_processed-network.csv`
- `_stats.csv`
- `_summary.csv`
- `epr_results.csv`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [evaluate_epr.py](evaluate_epr.py)
