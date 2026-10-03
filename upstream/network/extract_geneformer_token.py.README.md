# `upstream/network/extract_geneformer_token.py`

File-level notes generated from the current server source by static inspection.

## Dependencies

Imports found in the source (standard library and external modules): `argparse`, `datetime`, `numpy`, `os`, `pandas`, `pathlib`, `pickle`, `sklearn`, `sys`, `torch`, `transformers`.

## Defined interfaces

`parse_args`, `load_geneformer_embedding_df`, `extract_dataset_name`, `process_single_pair`, `main`

## Command-line parameters

- `--input-root`: Input root with CHIP/Non_CHIP/STRING folders.
- `--output-root`: Output directory for TSV + run records.
- `--model-dir`: Geneformer model directory (e.g. .../Geneformer/default/12L).
- `--dict-dir`: Geneformer dict directory containing token_dictionary.pkl etc.
- `--folders`: Folders to process. Default: ["CHIP"].

## Invocation

Run `python upstream/network/extract_geneformer_token.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `-ExpressionData.csv`
- `.tsv`
- `Geneformerrun_records.csv`
- `_chip_matched-ExpressionData.csv`
- `_processed-ExpressionData.csv`
- `gene_name_id_dict.pkl`
- `token_dictionary.pkl`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [extract_geneformer_token.py](extract_geneformer_token.py)
