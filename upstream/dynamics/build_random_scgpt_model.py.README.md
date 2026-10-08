# `upstream/dynamics/build_random_scgpt_model.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

Xavier/Glorot-random scGPT builder used by the Fig. 6a seeded control.

Only model construction lives here. Evaluation parameters come from
``run_scgpt_gene_results.py`` via ``run_random_scgpt_seeded.py``.

Linear and embedding weights use Xavier normal initialization. LayerNorm scales
and biases are initialized to one and zero. Native MultiheadAttention packed
QKV weights (or separate Q/K/V weights) are explicitly reinitialized, rather
than retaining the constructor's initialization. Pretrained loading uses the
same fail-closed checkpoint validation as the formal dynamic evaluator.

## Dependencies

Imports found in the source (standard library and external modules): `__future__`, `json`, `os`, `pathlib`, `scgpt`, `sys`, `torch`, `warnings`.

## Defined interfaces

`init_weights_xavier_normal`, `build_model`

## Invocation

This file does not declare an explicit `__main__` guard. Inspect top-level execution and call sites before importing or running it.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `/mnt/10T/yzn/benchmark_GRN/pre_scgpt/scGPT`
- `/mnt/10T/yzn/benchmark_GRN/pre_scgpt/scGPT/scgpt_human`
- `args.json`
- `best_model.pt`
- `vocab.json`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [build_random_scgpt_model.py](build_random_scgpt_model.py)
