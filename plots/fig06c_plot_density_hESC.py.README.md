# `plots/fig06c_plot_density_hESC.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

Plot Fig. 6c: hESC GRN propagation vs edge density, with per-point error bars.

Observed points are the median Spearman rho over the first eight refinement lags
(same definition as Methods). Error bars show the sample s.d. (ddof=1) of those
eight lag-wise Spearman values. Grey shading remains the pooled rewired-null
5th--95th percentile (3 representations x 200 rewirings = 600).

Reads archived propagation JSON under scGRN-Bench/FBplot/fig6/raw_rewiring/ and
writes figures + source CSV under paper-code/outputs/fig06c/.

## Dependencies

Imports found in the source (standard library and external modules): `__future__`, `argparse`, `json`, `matplotlib`, `numpy`, `pandas`, `pathlib`.

## Defined interfaces

`parse_args`, `load_observed_with_lag_errors`, `load_pooled_null`, `plot_figure`, `main`

## Command-line parameters

- `--json-dir`
- `--plot-data`
- `--outdir`
- `--error`: Error bars on observed points: lag sample s.d. (default), SEM, or IQR.

## Invocation

Run `python plots/fig06c_plot_density_hESC.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `.json`
- `.pdf`
- `.png`
- `/mnt/10T/yzn/scGRN-Bench/FBplot/fig6/plot_data.csv`
- `/mnt/10T/yzn/scGRN-Bench/FBplot/fig6/raw_rewiring`
- `fig06c_density_with_lag_errorbars_source_data.csv`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [fig06c_plot_density_hESC.py](fig06c_plot_density_hESC.py)
