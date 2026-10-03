# `upstream/baselines/grnboost/run_grnboost.py`

File-level notes generated from the current server source by static inspection.

## Dependencies

Imports found in the source (standard library and external modules): `arboreto`, `numpy`, `os`, `pandas`, `sklearn`.

## Defined interfaces

`extract_tf_list`, `load_expression_data`, `run_grnboost_inference`, `calculate_aupr_ratio`, `calculate_epr`, `main`

## Invocation

Run `python upstream/baselines/grnboost/run_grnboost.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `/grnboost_STR_summary.csv`
- `/grnboost_aupr_epr_summary.csv`
- `/mnt/10T/yzn/benchmark_GRN/input_process/STRING`
- `/{dataset}_processed-ExpressionData.csv`
- `/{dataset}_processed-network.csv`
- `_grnboost_result.tsv`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [run_grnboost.py](run_grnboost.py)
