# `upstream/network/export_scgpt_attention_heads.py`

File-level notes generated from the current server source by static inspection.

## Dependencies

Imports found in the source (standard library and external modules): `anndata`, `argparse`, `einops`, `extract_scgpt_hidden`, `json`, `numpy`, `os`, `pandas`, `pathlib`, `scanpy`, `scgpt`, `scipy`, `sys`, `torch`, `tqdm`, `utils_heads`, `warnings`.

## Defined interfaces

`parse_args`, `main`

## Command-line parameters

- `--data-type`
- `--dataset`
- `--input-root`
- `--output-root`
- `--scgpt-repo-dir`
- `--model-dir`
- `--batch-size`
- `--target-layer`
- `--head-indices`: Comma-separated head indices to export, e.g. '0,3,7'. Default 'all' exports every head.

## Invocation

Run `python upstream/network/export_scgpt_attention_heads.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `-ExpressionData.csv`
- `.tsv`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [export_scgpt_attention_heads.py](export_scgpt_attention_heads.py)
