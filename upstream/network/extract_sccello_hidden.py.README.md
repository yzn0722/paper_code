# `upstream/network/extract_sccello_hidden.py`

File-level notes generated from the current server source by static inspection.

## Dependencies

Imports found in the source (standard library and external modules): `argparse`, `datetime`, `json`, `numpy`, `os`, `pandas`, `pathlib`, `pickle`, `sc_foundation_evals`, `sys`, `torch`, `tqdm`, `traceback`, `warnings`.

## Defined interfaces

`parse_args`, `set_seed`, `_safe`, `extract_dataset_name`, `read_expression_matrix`, `get_pad_id`, `_get_hidden`, `load_sccello_resources`, `build_gene_sequences`, `extract_hidden_embeddings`, `compute_and_save_all_cosine_edges_stream`, `compute_and_save_topk_cosine_edges`, `process_single_dataset`, `main`

## Command-line parameters

- `--input-root`
- `--output-root`
- `--model-dir`
- `--dict-dir`
- `--repo-root`: Optional repo root to append to PYTHONPATH for sc_foundation_evals.
- `--folders`
- `--batch-size`
- `--seed`

## Invocation

Run `python upstream/network/extract_sccello_hidden.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `*_chip_matched-ExpressionData.csv`
- `*_processed-ExpressionData.csv`
- `-ExpressionData.csv`
- `.tsv`
- `_chip_matched-ExpressionData.csv`
- `_processed-ExpressionData.csv`
- `_processing_summary.csv`
- `gene_name_id_dict.pkl`
- `gene_name_id_dict.pkl + token_dictionary.pkl`
- `run_params.json`
- `token_dictionary.pkl`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [extract_sccello_hidden.py](extract_sccello_hidden.py)
