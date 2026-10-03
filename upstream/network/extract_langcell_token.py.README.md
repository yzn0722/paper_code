# `upstream/network/extract_langcell_token.py`

File-level notes generated from the current server source by static inspection.

## Dependencies

Imports found in the source (standard library and external modules): `argparse`, `datetime`, `numpy`, `os`, `pandas`, `pathlib`, `pickle`, `sklearn`, `sys`, `torch`, `transformers`.

## Defined interfaces

`load_langcell_model_and_dict`, `parse_args`, `extract_dataset_name`, `process_single_pair`, `main`

## Command-line parameters

- `--input-root`: Input root with CHIP/Non_CHIP/STRING folders.
- `--output-root`: Output directory for TSV + run records.
- `--parent-model-dir`: Parent weights directory containing LangCell/ and Geneformer/dicts/.
- `--folders`: Folders to process. Default: ["CHIP"].

## Invocation

Run `python upstream/network/extract_langcell_token.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `-ExpressionData.csv`
- `.tsv`
- `LangCellrun_records.csv`
- `_chip_matched-ExpressionData.csv`
- `_processed-ExpressionData.csv`
- `gene_name_id_dict.pkl`
- `token_dictionary.pkl`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [extract_langcell_token.py](extract_langcell_token.py)
