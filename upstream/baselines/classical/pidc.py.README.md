# `upstream/baselines/classical/pidc.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

Python reimplementation of PIDC (Chan et al., Cell Systems 2017)
matching NetworkInference.jl / BEELINE:

  1) Discretize expression (uniform_width by default)
  2) Pairwise MI + specific information
  3) PUC scores over gene triples (proportional unique contribution)
  4) Context weighting via Gamma CDF (PIDC) → undirected edge weights

Reference:
  https://github.com/Tchanders/NetworkInference.jl
  BEELINE Algorithms/PIDC/runPIDC.jl

## Dependencies

Imports found in the source (standard library and external modules): `__future__`, `numba`, `numpy`, `pandas`, `scipy`, `typing`.

## Defined interfaces

`_safe_log`, `discretize_uniform_width`, `discretize_uniform_count`, `joint_counts_2d`, `ml_probs`, `mutual_information`, `specific_information`, `_accumulate_puc`, `apply_pidc_context`, `infer_pidc`, `load_expression_csv`

## Invocation

This file does not declare an explicit `__main__` guard. Inspect top-level execution and call sites before importing or running it.

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [pidc.py](pidc.py)
