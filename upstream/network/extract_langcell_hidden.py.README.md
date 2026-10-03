# `upstream/network/extract_langcell_hidden.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

LangCell hidden-state gene embeddings -> cosine edges (Geneformer-aligned, batch)

 Geneformer :
   Symbol -> gene_name_id_dict.pkl -> ENSG -> token_dictionary.pkl

Dataset 1  TSV :
   {MODEL_NAME}_{DATASET}.tsv   (:Gene1, Gene2, EdgeWeight;;)

:
   run_params.json

 embedding (embedding )

## Dependencies

Imports found in the source (standard library and external modules): `argparse`, `datetime`, `json`, `numpy`, `os`, `pandas`, `pathlib`, `pickle`, `torch`, `tqdm`, `traceback`, `transformers`, `warnings`.

## Defined interfaces

`parse_args`, `set_seed`, `_safe`, `extract_dataset_name`, `read_expression_matrix`, `load_token_dictionary`, `load_gene_name_id_dict`, `get_pad_id`, `match_genes_to_tokens_like_geneformer`, `_get_hidden`, `build_sequences`, `compute_and_save_all_cosine_edges_stream`, `compute_and_save_topk_cosine_edges`, `process_single_dataset`, `main`

## Command-line parameters

- `--input-root`
- `--output-root`
- `--dict-path`
- `--langcell-model-path`
- `--folders`
- `--batch-size`
- `--seed`

## Invocation

Run `python upstream/network/extract_langcell_hidden.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `*_chip_matched-ExpressionData.csv`
- `*_processed-ExpressionData.csv`
- `-ExpressionData.csv`
- `.tsv`
- `_chip_matched-ExpressionData.csv`
- `_processed-ExpressionData.csv`
- `gene_name_id_dict.pkl`
- `run_params.json`
- `token_dictionary.pkl`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [extract_langcell_hidden.py](extract_langcell_hidden.py)
