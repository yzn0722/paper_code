# `upstream/dynamics/run_cross_dataset_grn_density.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

Compute and plot cross-dataset GRN/dynamics coupling for scGPT.

The calculation matches the hESC density plot: absolute expression change at
iteration t is propagated through an incoming-normalized GRN and compared with
the absolute change at t+1 using Spearman correlation.  The reported transient
score is the median across the first eight available lags.

## Dependencies

Imports found in the source (standard library and external modules): `__future__`, `argparse`, `json`, `matplotlib`, `numpy`, `pandas`, `pathlib`, `run_weighted_grn_propagation`, `sys`.

## Defined interfaces

`parse_args`, `find_case_insensitive`, `expression_path`, `score_dataset`, `hesc_reference_rows`, `plot_bar`, `main`

## Command-line parameters

- `--benchmark-root`
- `--trajectory-root`
- `--outdir`
- `--datasets`
- `--transient-lags`
- `--bar-density`
- `--n-null`: Degree-preserving rewired networks per representation at bar density.
- `--seed`
- `--plot-only`: Render the chart from existing observed and rewired-null CSV files.
- `--use-hesc-reference`
- `--no-use-hesc-reference`

## Invocation

Run `python upstream/dynamics/run_cross_dataset_grn_density.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `*.tsv`
- `.pdf`
- `.png`
- `/mnt/10T/yzn/benchmark_GRN`
- `/mnt/10T/yzn/scGRN-Bench/outputs/dynamic_grn_cross_dataset`
- `/mnt/10T/yzn/scGRN-Bench/outputs/grn_density_cross_dataset`
- `_chip_matched-ExpressionData.csv`
- `early_mean_trajectory.npy`
- `grn_transient_spearman_cross_dataset_all_densities.csv`
- `grn_transient_spearman_cross_dataset_rewired_null.csv`
- `vocab.json`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [run_cross_dataset_grn_density.py](run_cross_dataset_grn_density.py)
