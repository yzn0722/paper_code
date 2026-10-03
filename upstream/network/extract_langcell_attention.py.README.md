# `upstream/network/extract_langcell_attention.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

LangCell Attention - (CHIP/Non_CHIP/STRING)
,CSV,

## Dependencies

Imports found in the source (standard library and external modules): `datetime`, `json`, `numpy`, `os`, `pandas`, `pathlib`, `pickle`, `scanpy`, `scipy`, `shutil`, `sys`, `torch`, `tqdm`, `transformers`, `warnings`.

## Defined interfaces

`Pooler`, `GeneExpressionDataset`, `DataCollatorForCellClassification`, `reverse_permute`, `tokenize_cell`, `LangCellAttentionExtractor`, `main`

## Invocation

Run `python upstream/network/extract_langcell_attention.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `-ExpressionData.csv`
- `.tsv`
- `/mnt/10T/yzn/benchmark_GRN/input_process`
- `/mnt/10T/yzn/benchmark_GRN/model/output_att500/langcell`
- `/mnt/10T/yzn/benchmark_GRN/model/weights/Geneformer/dicts`
- `/mnt/10T/yzn/benchmark_GRN/model/weights/LangCell`
- `gene_median_dictionary.pkl`
- `gene_name_id_dict.pkl`
- `langcell-all-params.csv`
- `token_dictionary.pkl`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [extract_langcell_attention.py](extract_langcell_attention.py)
