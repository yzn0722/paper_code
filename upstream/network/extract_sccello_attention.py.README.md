# `upstream/network/extract_sccello_attention.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

scCello Attention - (CHIP/Non_CHIP+CSV)
,DatasetCSV

## Dependencies

Imports found in the source (standard library and external modules): `argparse`, `collections`, `datasets`, `geneformer`, `model_prototype_contrastive`, `numpy`, `os`, `pandas`, `pathlib`, `pickle`, `scanpy`, `scipy`, `shutil`, `sys`, `time`, `torch`, `tqdm`.

## Defined interfaces

`parse_args`, `SimpleCollator`, `reverse_permute`, `add_labels_and_indices`, `convert_to_lists`

## Command-line parameters

- `data_type`
- `dataset`
- `--input-root`
- `--output-root`
- `--model-path`
- `--dict-dir`
- `--sccello-repo-dir`: Path containing sccello/src.
- `--batch-size`
- `--num-workers`
- `--target-layer`

## Invocation

This file does not declare an explicit `__main__` guard. Inspect top-level execution and call sites before importing or running it.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `-ExpressionData.csv`
- `-all-params.csv`
- `-gene_attention_edges.tsv`
- `/gene_median_dictionary.pkl`
- `/gene_name_id_dict.pkl`
- `/token_dictionary.pkl`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [extract_sccello_attention.py](extract_sccello_attention.py)
