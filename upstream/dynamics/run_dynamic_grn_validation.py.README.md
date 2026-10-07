# `upstream/dynamics/run_dynamic_grn_validation.py`

Resource-guarded scGPT dynamics validation: early/middle/late trajectories,
convergence metrics, GRN lag scores against rewired controls and optional TF
perturbation probes. Outputs retain group means rather than all cell trajectories.
The default invocation performs preflight; `--execute` runs inference.

## Dynamics protocol

`--ema-alpha` is the coefficient on the previous state and defaults to **0.9**:
`next = 0.9 * previous + 0.1 * mlm_output`.

The repository's formal scGPT loader and native per-cell binning helper replace
the unavailable external `run_grn_perturbation_probes.ScgptBundle` dependency.
Uppercased vocabulary-mapped genes are binned separately in each cell, with
checkpoint `n_bins` (51 if absent). `--scgpt-bin-log1p` is disabled by default;
leave it disabled to reproduce Methods. Unmapped genes are zero-filled, assigned
padding tokens and frozen, as is CLS. Binning is applied once before iterative
continuous EMA refinement. The result JSON records preprocessing and EMA settings.

Early/middle/late groups use pseudotime quantiles (`--pt-quantile 0.2`); the script
refines each sampled cell and saves its group's mean at every iteration. Defaults
are 32 iterations, up to 16 cells per group, batch size 4 and seed 0. These settings
are configurable; they are not inferred from a saved manuscript figure.

## Inputs, dependencies and invocation

Requires NumPy, pandas, SciPy, PyTorch and a compatible scGPT installation,
checkpoint and this repository's dynamics/checkpoint modules. Supply:

- `--expr-root`: expression and CHIP CSVs in its `CHIP` directory.
- `--pt-root`: `<dataset>/PseudoTime.csv`.
- `--scgpt-model-dir`: checkpoint `args.json`, `vocab.json`, `best_model.pt`.
- `--scgpt-repo-dir`: scGPT source checkout.
- Optional `--chip-network` overrides the default CHIP CSV.

```sh
python upstream/dynamics/run_dynamic_grn_validation.py \
  --dataset hESC --expr-root /path/input_process --pt-root /path/PseudoTime \
  --scgpt-model-dir /path/scgpt_human --scgpt-repo-dir /path/scGPT \
  --outdir outputs/dynamic_grn_validation_binned_ema09 \
  --ema-alpha 0.9 --device cuda --execute
```

Omit `--execute` for preflight. `--self-test` runs dependency-light metric checks
(checkpoint path arguments remain syntactically required). `--scgpt-legacy-pt`
is retained as a deprecated compatibility flag; pseudotime uses the shared reader.
`--string-network` is retained for CLI compatibility and is not loaded.

## Outputs and verification

Each dataset output folder contains `early_mean_trajectory.npy`,
`middle_mean_trajectory.npy`, `late_mean_trajectory.npy` and
`dynamic_grn_validation.json`. Use a new output folder when changing protocol;
older trajectories remain historical results and are not corrected by code edits.

CPU tests cover mapped binning, EMA, frozen values and actual execution/export
with a deterministic model. They do not verify real pretrained inference or
regenerate reported manuscript statistics.

Source: [run_dynamic_grn_validation.py](run_dynamic_grn_validation.py)
