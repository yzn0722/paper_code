# `upstream/network/extract_geneformer_attention.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

Geneformer attention extraction - batch mode (supports CHIP/Non_CHIP/STRING)
Unified output format with batch-level parameter logging.

## Dependencies

Imports found in the source (standard library and external modules): `argparse`, `datasets`, `datetime`, `geneformer`, `loompy`, `numpy`, `os`, `pandas`, `pathlib`, `pickle`, `scanpy`, `scipy`, `shutil`, `sys`, `time`, `torch`, `tqdm`, `transformers`.

## Defined interfaces

`parse_args`, `rank_normalize_fixed`, `reverse_permute_fixed`, `collate_fn`

## Command-line parameters

- `data_type`
- `dataset`
- `--weights-root`: Root directory containing Geneformer/.
- `--input-root`: TFs+500 expression root (CHIP/Non_CHIP/STRING). Use input_process1000 only for the TFs+1000 ablation.
- `--output-root`: Attention TSV output root for the TFs+500 setting (output_att500).
- `--model-version`: Geneformer model version folder name.

## Invocation

This file does not declare an explicit `__main__` guard. Inspect top-level execution and call sites before importing or running it.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `-ExpressionData.csv`
- `.tsv`
- `/gene_median_dictionary.pkl`
- `/gene_name_id_dict.pkl`
- `/mnt/10T/yzn/benchmark_GRN/input_process`
- `/mnt/10T/yzn/benchmark_GRN/model/output_att500/geneformer`
- `/mnt/10T/yzn/benchmark_GRN/model/weights`
- `/token_dictionary.pkl`
- `geneformer-all-params.csv`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [extract_geneformer_attention.py](extract_geneformer_attention.py)
