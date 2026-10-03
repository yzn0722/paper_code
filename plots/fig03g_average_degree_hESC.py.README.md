# `plots/fig03g_average_degree_hESC.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

Grouped bar chart: Average Degree

## Dependencies

Imports found in the source (standard library and external modules): `__future__`, `argparse`, `fig2_palette`, `matplotlib`, `networkx`, `numpy`, `pandas`, `pathlib`.

## Defined interfaces

`normalize_edges`, `detect_weight_col`, `load_string`, `resolve_pred_path`, `load_filtered_pred`, `calc_avg_degree`, `calc_string_metric`, `compute_table`, `plot_bar`, `main`

## Command-line parameters

- `--dataset`
- `--top-edges`

## Invocation

Run `python plots/fig03g_average_degree_hESC.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `*.tsv`
- `.csv`
- `.pdf`
- `.tsv`
- `/mnt/10T/yzn/benchmark_GRN`
- `_avg_degree_6models.pdf`
- `_processed-network.csv`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [fig03g_average_degree_hESC.py](fig03g_average_degree_hESC.py)
