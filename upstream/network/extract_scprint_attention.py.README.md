# `upstream/network/extract_scprint_attention.py`

File-level notes generated from the current server source by static inspection.

## Dependencies

Imports found in the source (standard library and external modules): `argparse`, `datetime`, `gc`, `inspect`, `mygene`, `os`, `pandas`, `pickle`, `re`, `scanpy`, `scipy`, `scprint`, `time`, `torch`, `traceback`, `types`.

## Defined interfaces

`parse_args`, `write_txt`, `datetime_now`, `_normalize_state_dict_keys`, `_is_ensg`, `load_env_genes_from_token_pkl`, `clean_symbol`, `map_symbols_to_ensembl`, `csv_to_adata`, `disable_bias_runtime`, `load_model_with_embedding_alignment_fp32_normal`, `infer_grn_fp32_noamp_slice_attn`, `main`

## Command-line parameters

- `--ckpt`: Path to scPRINT .ckpt file.
- `--token-pkl`: Path to token_dictionary.pkl.
- `--expr-csv`: Path to ExpressionData.csv (genes x cells).
- `--out-prefix`: Output prefix (directory will be created).
- `--species`: Species ontology term id.
- `--cell-type-name`: Cell type label stored in adata.obs.
- `--batch-size`: GNInfer batch size.
- `--topk`: TopK edges per gene (when filtration uses top-k).
- `--filtration`: Edge filtration.
- `--head-agg`: Attention head aggregation.
- `--preprocess`: Preprocess.
- `--forward-mode`: Forward mode (keep 'none' unless needed).
- `--ckpt-gene-emb-offset`: Embedding row offset (if ckpt has specials).

## Invocation

Run `python upstream/network/extract_scprint_attention.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `_grn.h5ad`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [extract_scprint_attention.py](extract_scprint_attention.py)
