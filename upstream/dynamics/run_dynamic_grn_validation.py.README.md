# `upstream/dynamics/run_dynamic_grn_validation.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

Resource-guarded validation of scGPT iterative dynamics and GRN coupling.

This script addresses four questions in one reproducible protocol:
1. Does a long iteration converge, and is its fixed point close to late cells?
2. Are observed late cells approximately stationary under the same operator?
3. Do middle-pseudotime cells continue along the early-to-late axis without
   collapsing to one common mean state?
4. Are successive changes and TF perturbation responses more coherent on the
   observed directed GRN than on degree-preserving rewired controls?

The default invocation is a read-only preflight. Pass --execute explicitly to
load scGPT and run inference. Outputs contain group-mean trajectories rather
than all cell-by-iteration tensors, keeping memory bounded.

## Dependencies

Imports found in the source (standard library and external modules): `__future__`, `argparse`, `csv`, `json`, `math`, `numpy`, `os`, `pandas`, `pathlib`, `run_grn_perturbation_probes`, `scipy`, `sys`, `time`, `torch`, `typing`.

## Defined interfaces

`parse_args`, `_json_float`, `safe_spearman`, `degree_preserving_rewire`, `trajectory_metrics`, `edge_lag_score`, `grn_null_test`, `estimate_forward_passes`, `resolve_input_paths`, `preflight`, `select_indices`, `run_group`, `choose_probe_tfs`, `tf_response_probe`, `execute`, `self_test`, `main`

## Command-line parameters

- `--dataset`
- `--outdir`
- `--expr-root`
- `--pt-root`
- `--chip-network`
- `--string-network`: Optional STRING network path (kept for bundle compatibility).
- `--grn-tsv`: Optional predicted GRN; CHIP is always the primary graph.
- `--scgpt-model-dir`
- `--scgpt-repo-dir`
- `--device`
- `--execute`: Actually load the model and run; otherwise preflight only.
- `--self-test`: Run dependency-light metric tests and exit.
- `--gen-iters`
- `--max-cells`: Maximum cells per early/middle/late start group.
- `--batch-size`
- `--ema-alpha`
- `--pt-quantile`
- `--top-percent`
- `--convergence-tol`
- `--convergence-patience`
- `--n-rewired`
- `--seed`
- `--n-tf-probes`
- `--probe-iters`
- `--perturb-delta`
- `--n-random-target-sets`: Random label sets for the pure-random target enrichment sensitivity test.
- `--max-forward-passes`
- `--memory-limit-gb`
- `--cpu-threads`: PyTorch CPU thread cap for this process.
- `--scgpt-bin-log1p`
- `--scgpt-legacy-pt`
- `--top-grn-edges`
- `--print-every`

## Invocation

Run `python upstream/dynamics/run_dynamic_grn_validation.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `/mnt/10T/yzn/benchmark_GRN/PseudoTime`
- `/mnt/10T/yzn/benchmark_GRN/input_process`
- `PseudoTime.csv`
- `_chip_matched-ExpressionData.csv`
- `_chip_matched-network.csv`
- `_mean_trajectory.npy`
- `best_model.pt`
- `dynamic_grn_validation.json`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [run_dynamic_grn_validation.py](run_dynamic_grn_validation.py)
