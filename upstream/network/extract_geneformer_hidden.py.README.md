# `upstream/network/extract_geneformer_hidden.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

Geneformer hidden-state -> cosine edges (LangCell-aligned IO rules)

: CHIP/Non_CHIP/STRING , ExpressionData.csv
   - CHIP: *_chip_matched-ExpressionData.csv
   - others: *_processed-ExpressionData.csv

: TSV
   {MODEL_NAME}_{DATASET}.tsv  (columns: Gene1, Gene2, EdgeWeight; directed; no self-loop)

:run_params.json

 embedding (Generated)

## Dependencies

Imports found in the source (standard library and external modules): `argparse`, `collections`, `datetime`, `json`, `numpy`, `os`, `pandas`, `pathlib`, `pickle`, `torch`, `tqdm`, `traceback`, `transformers`, `typing`, `warnings`.

## Defined interfaces

`parse_args`, `set_seed`, `_safe`, `extract_dataset_name`, `load_token_dictionary`, `load_gene_name_id_dict`, `match_genes_to_tokens`, `build_cell_sequences`, `GeneSequenceDataset`, `collate_fn`, `extract_hidden_embeddings`, `compute_and_save_all_cosine_edges_stream`, `compute_and_save_topk_cosine_edges`, `process_single_dataset`, `main`

## Command-line parameters

- `--model-dir`
- `--dict-dir`
- `--input-root`
- `--output-root`
- `--folders`
- `--batch-size`
- `--seed`

## Invocation

Run `python upstream/network/extract_geneformer_hidden.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `*_chip_matched-ExpressionData.csv`
- `*_processed-ExpressionData.csv`
- `-ExpressionData.csv`
- `.tsv`
- `_chip_matched-ExpressionData.csv`
- `_processed-ExpressionData.csv`
- `gene_name_id_dict.pkl`
- `gene_name_id_dict.pkl + token_dictionary.pkl`
- `processing_summary.json`
- `run_params.json`
- `token_dictionary.pkl`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [extract_geneformer_hidden.py](extract_geneformer_hidden.py)
