# `upstream/network/extract_scfoundation_token.py`

File-level notes generated from the current server source by static inspection.

## Dependencies

Imports found in the source (standard library and external modules): `argparse`, `copy`, `datetime`, `json`, `numpy`, `os`, `pandas`, `pathlib`, `pickle`, `sklearn`, `sys`, `time`, `torch`, `tqdm`, `traceback`, `warnings`, `yaml`.

## Defined interfaces

`parse_args`, `extract_dataset_name`, `save_run_statistics`, `GeneEmbeddingProcessor`, `process_single_pair`, `main`

## Command-line parameters

- `--input-root`: Input root with CHIP/Non_CHIP/STRING subfolders.
- `--output-root`: Output directory for TSV + run logs.
- `--ckpt-path`: scFoundation models.ckpt path.
- `--vocab-path`: OS_scRNA_gene_index.19264.tsv path.
- `--folders`: Folders to process. Default: ["CHIP"].

## Invocation

Run `python upstream/network/extract_scfoundation_token.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `-ExpressionData.csv`
- `.tsv`
- `_chip_matched-ExpressionData.csv`
- `_processed-ExpressionData.csv`
- `scFoundation_run_statistics.csv`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [extract_scfoundation_token.py](extract_scfoundation_token.py)
