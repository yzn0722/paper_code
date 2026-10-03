# `plots/fig06d_cross_dataset_grn_propagation.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

Compute and plot cross-dataset GRN/dynamics coupling for scGPT.

The calculation matches the hESC density plot: absolute expression change at
iteration t is propagated through an incoming-normalized GRN and compared with
the absolute change at t+1 using Spearman correlation.  The reported transient
score is the median across the first eight available lags.

## Dependencies

Imports found in the source (standard library and external modules): `__future__`, `argparse`, `fig06c_grn_propagation_hESC`, `json`, `matplotlib`, `numpy`, `pandas`, `pathlib`, `sys`.

## Defined interfaces

`parse_args`, `find_case_insensitive`, `expression_path`, `score_dataset`, `load_hesc_results_csv`, `plot_bar`, `main`

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
- `--hesc-results-csv`: CSV of computed hESC observed and per-rewiring scores.
- `--refresh-hesc-only`: Replace hESC rows in existing result CSVs from --hesc-results-csv and redraw.

## Invocation

Run `python plots/fig06d_cross_dataset_grn_propagation.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `*.tsv`
- `.pdf`
- `.png`
- `/mnt/10T/yzn/benchmark_GRN`
- `/mnt/10T/yzn/scGRN-Bench/FBplot/fig6/raw_rewiring/rewiring_null_scores_for_plot.csv`
- `/mnt/10T/yzn/scGRN-Bench/outputs/dynamic_grn_cross_dataset`
- `/mnt/10T/yzn/scGRN-Bench/outputs/grn_density_cross_dataset`
- `_chip_matched-ExpressionData.csv`
- `early_mean_trajectory.npy`
- `grn_transient_spearman_cross_dataset_all_densities.csv`
- `grn_transient_spearman_cross_dataset_rewired_null.csv`
- `vocab.json`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [fig06d_cross_dataset_grn_propagation.py](fig06d_cross_dataset_grn_propagation.py)
