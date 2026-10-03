# `upstream/baselines/regformer/prepare_beeline_h5ad.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

Convert BEELINE STRING ExpressionData CSVs to AnnData h5ad for RegFormer.

## Dependencies

Imports found in the source (standard library and external modules): `__future__`, `anndata`, `numpy`, `pandas`, `pathlib`.

## Defined interfaces

`convert_one`, `main`

## Invocation

Run `python upstream/baselines/regformer/prepare_beeline_h5ad.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `.h5ad`
- `/mnt/10T/yzn/RegFormer/data`
- `/mnt/10T/yzn/benchmark_GRN/input_process/STRING`
- `_processed-ExpressionData.csv`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [prepare_beeline_h5ad.py](prepare_beeline_h5ad.py)
