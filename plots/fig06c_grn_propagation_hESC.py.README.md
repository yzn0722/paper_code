# `plots/fig06c_grn_propagation_hESC.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

Test whether iterative expression changes propagate through weighted GRNs.

For unsigned attention weights, this evaluates
    predicted |dX(t+1)| = W.T @ |dX(t)|
against observed |dX(t+1)|. Real networks are compared with directed
configuration-model rewiring (exact in/out stub counts) and weight shuffling.

No model is loaded. The script only reads saved mean trajectories and a GRN TSV.

## Dependencies

Imports found in the source (standard library and external modules): `__future__`, `argparse`, `hashlib`, `json`, `math`, `numpy`, `pandas`, `pathlib`, `scipy`, `time`, `typing`.

## Defined interfaces

`parse_args`, `file_sha256`, `finite`, `corr`, `score_vectors`, `normalize_incoming`, `propagate`, `score_trajectory`, `summarize_window`, `configuration_rewire`, `empirical_p`, `compare_nulls`, `load_inputs`, `load_grn`, `preflight`, `run`, `self_test`

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

Run `python plots/fig06c_grn_propagation_hESC.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `_mean_trajectory.npy`
- `weighted_grn_propagation.json`
- `weighted_grn_propagation_curves.csv`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Fig. 6c propagation was recomputed on saved hESC trajectories on 2026-10-06; the command does not load or rerun the model.

Source: [fig06c_grn_propagation_hESC.py](fig06c_grn_propagation_hESC.py)

## Fig. 6c direction and provenance

Primary direction is `query_to_key_primary` (Gene1 to Gene2), with
`key_to_query_sensitivity` reserved for the reverse analysis. `--primary-only`
omits the reverse analysis. Do not relabel a legacy key-to-query report.
Reports include trajectory, expression, vocabulary, network and script hashes,
the base seed and transient-lag count. Both packaged copies of this calculator
use the same implementation.

Use `upstream/dynamics/run_fig06c_query_to_key.py` to recompute all three
representations and generate a plot with matching observed and rewired inputs.
