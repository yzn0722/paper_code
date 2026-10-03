# `upstream/network/extract_scgpt_attention.py`

File-level notes generated from the current server source by static inspection.

## Dependencies

Imports found in the source (standard library and external modules): `anndata`, `copy`, `datetime`, `einops`, `extract_scgpt_hidden`, `glob`, `json`, `numpy`, `os`, `pandas`, `pathlib`, `scanpy`, `scgpt`, `scipy`, `sys`, `torch`, `tqdm`, `warnings`.

## Defined interfaces

`init_scgpt_model`, `packed_qkv`, `get_all_expression_files`, `process_single_dataset`, `main`

## Invocation

Run `python upstream/network/extract_scgpt_attention.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `*_chip_matched-ExpressionData.csv`
- `*_processed-ExpressionData.csv`
- `.tsv`
- `/mnt/10T/yzn/benchmark_GRN/input_process`
- `/mnt/10T/yzn/benchmark_GRN/model/output_att500/scgpt`
- `/mnt/10T/yzn/benchmark_GRN/pre_scgpt/scGPT`
- `/mnt/10T/yzn/benchmark_GRN/pre_scgpt/scGPT/scgpt_human`
- `_run_parameters.csv`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [extract_scgpt_attention.py](extract_scgpt_attention.py)
