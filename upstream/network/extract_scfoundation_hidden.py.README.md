# `upstream/network/extract_scfoundation_hidden.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

scFoundation hidden -> cosine edges (/)

""( 910)Generated:
  -  19264 Model
  -  19264 embedding  embedding
  - 

## Dependencies

Imports found in the source (standard library and external modules): `argparse`, `datetime`, `json`, `load`, `numpy`, `os`, `pandas`, `pathlib`, `random`, `sys`, `torch`, `tqdm`, `traceback`, `warnings`.

## Defined interfaces

`parse_args`, `main_gene_selection`, `load_official_gene_list`, `set_seed`, `load_official_model`, `extract_dataset_name`, `read_expression_matrix_gene_by_cell`, `extract_gene_embedding_mean`, `compute_and_save_all_cosine_edges_stream`, `process_single_dataset`, `process_all_datasets`

## Command-line parameters

- `--input-root`
- `--output-root`
- `--scfoundation-root`
- `--model-path`
- `--vocab-path`
- `--batch-size`
- `--seed`
- `--device`

## Invocation

Run `python upstream/network/extract_scfoundation_hidden.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `*_chip_matched-ExpressionData.csv`
- `*_processed-ExpressionData.csv`
- `-ExpressionData.csv`
- `.tsv`
- `_chip_matched-ExpressionData.csv`
- `_processed-ExpressionData.csv`
- `_processing_summary.csv`
- `run_params.json`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [extract_scfoundation_hidden.py](extract_scfoundation_hidden.py)
