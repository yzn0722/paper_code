# `upstream/network/extract_scgpt_hidden.py`

File-level notes generated from the current server source by static inspection.

## Dependencies

Imports found in the source (standard library and external modules): `argparse`, `copy`, `datetime`, `json`, `numpy`, `os`, `pandas`, `pathlib`, `scgpt`, `sys`, `torch`, `tqdm`, `traceback`, `typing`, `warnings`.

## Defined interfaces

`_env_or`, `parse_args`, `init_scgpt_model`, `extract_dataset_name`, `clear_gpu_cache`, `read_expression_matrix`, `build_gene_sequences`, `extract_hidden_embeddings`, `compute_and_save_all_cosine_edges_stream`, `process_single_dataset`, `main`

## Command-line parameters

- `--input-root`: Input root directory containing CHIP/Non_CHIP/STRING subfolders with *ExpressionData.csv files.
- `--output-root`: Output root directory. Each dataset writes one {MODEL_NAME}_{DATASET}.tsv plus run_params.json.
- `--model-dir`: scGPT model directory containing vocab.json, args.json, best_model.pt.
- `--folders`: Subfolders under input-root to process (e.g. CHIP Non_CHIP STRING). Default: ["CHIP"].
- `--seed`: Random seed.
- `--batch-size`: Forward batch size.
- `--seq-topk-genes`: Top-K expressed genes per cell to build sequences.
- `--max-seq-len`: Maximum sequence length.
- `--n-cells`: Use only N sampled cells (default: all).
- `--use-log1p`: Apply log1p to expression values before ranking.
- `--no-save-all-edges`: Disable all-edges export (use topK per gene).
- `--topk-per-gene`: When not saving all edges, topK edges per Gene1.
- `--clear-cache-every`: Clear CUDA cache every N batches.
- `--evaluation-layout`: Write output-root/scgpt_hidden/scGPT_DATASET.tsv for evaluate_aupr.py/evaluate_epr.py; requires one --folders value.

## Invocation

Run `python upstream/network/extract_scgpt_hidden.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `*_chip_matched-ExpressionData.csv`
- `*_processed-ExpressionData.csv`
- `-ExpressionData.csv`
- `.csv`
- `.tsv`
- `_chip_matched-ExpressionData.csv`
- `_processed-ExpressionData.csv`
- `_processing_summary.csv`
- `_run_params.json`
- `args.json`
- `best_model.pt`
- `run_params.json`
- `vocab.json`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [extract_scgpt_hidden.py](extract_scgpt_hidden.py)
