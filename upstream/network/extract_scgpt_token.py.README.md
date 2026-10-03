# `upstream/network/extract_scgpt_token.py`

File-level notes generated from the current server source by static inspection.

## Dependencies

Imports found in the source (standard library and external modules): `argparse`, `copy`, `datetime`, `json`, `numpy`, `os`, `pandas`, `pathlib`, `pickle`, `scanpy`, `scgpt`, `sklearn`, `sys`, `time`, `torch`, `tqdm`, `traceback`, `warnings`, `yaml`.

## Defined interfaces

`parse_args`, `save_run_statistics`, `extract_dataset_name`, `GeneEmbeddingProcessor`, `preload_scgpt_embeddings`, `process_single_pair`, `main`

## Command-line parameters

- `--input-root`: Input root with CHIP/Non_CHIP/STRING subfolders.
- `--output-root`: Output directory for TSV + run logs.
- `--model-dir`: scGPT model dir containing vocab.json and best_model.pt.
- `--folders`: Folders to process. Default: ["CHIP"].

## Invocation

Run `python upstream/network/extract_scgpt_token.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `-ExpressionData.csv`
- `.tsv`
- `_chip_matched-ExpressionData.csv`
- `_processed-ExpressionData.csv`
- `best_model.pt`
- `scGPT_run_statistics.csv`
- `vocab.json`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [extract_scgpt_token.py](extract_scgpt_token.py)
