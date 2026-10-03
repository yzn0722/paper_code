# `upstream/dynamics/run_weighted_grn_propagation.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

Test whether iterative expression changes propagate through weighted GRNs.

For unsigned attention weights, this evaluates
    predicted |dX(t+1)| = W.T @ |dX(t)|
against observed |dX(t+1)|. Real networks are compared with directed
configuration-model rewiring (exact in/out stub counts) and weight shuffling.

No model is loaded. The script only reads saved mean trajectories and a GRN TSV.

## Dependencies

Imports found in the source (standard library and external modules): `__future__`, `argparse`, `json`, `math`, `numpy`, `pandas`, `pathlib`, `scipy`, `time`, `typing`.

## Defined interfaces

`parse_args`, `finite`, `corr`, `score_vectors`, `normalize_incoming`, `propagate`, `score_trajectory`, `summarize_window`, `configuration_rewire`, `empirical_p`, `compare_nulls`, `load_inputs`, `load_grn`, `preflight`, `run`, `self_test`

## Command-line parameters

- `--trajectory-dir`
- `--expression-csv`: Expression CSV supplying trajectory gene order.
- `--grn-tsv`
- `--network-name`
- `--vocab-json`: Optional scGPT vocab; excludes OOV/frozen genes.
- `--outdir`
- `--top-k`
- `--groups`
- `--n-null`
- `--transient-iters`
- `--top-fraction`
- `--seed`
- `--memory-limit-gb`
- `--execute`: Run after preflight; default is read-only preflight.
- `--primary-only`: Run query-to-key orientation only.
- `--self-test`

## Invocation

Run `python upstream/dynamics/run_weighted_grn_propagation.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `_mean_trajectory.npy`
- `weighted_grn_propagation.json`
- `weighted_grn_propagation_curves.csv`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [run_weighted_grn_propagation.py](run_weighted_grn_propagation.py)
