# `plots/fig03d_modularity_hESC.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

Plot grouped bar chart of modularity:
  6 models x 3 extraction types (emb500 / att500 / embhidden500)

Computation uses REAL predicted edges with STRING-based filtering:
  Gene1 in STRING(Gene1), Gene2 in STRING(Gene1 ∪ Gene2)
and keeps top-K edges (K = #STRING edges for that dataset by default).

## Dependencies

Imports found in the source (standard library and external modules): `__future__`, `argparse`, `fig2_palette`, `matplotlib`, `networkx`, `numpy`, `pandas`, `pathlib`, `typing`.

## Defined interfaces

`normalize_edges`, `detect_weight_col`, `load_string`, `resolve_pred_path`, `load_filtered_pred`, `calc_modularity`, `calc_string_modularity`, `compute_table`, `plot_bar`, `parse_args`, `main`

## Command-line parameters

- `--dataset`: Dataset name
- `--top-edges`: Top-K edges after filtering (0 means #STRING edges)
- `--out`: Output figure path (default: fig2/output/modularity_bar/<dataset>_modularity_6models_3extract.pdf)

## Invocation

Run `python plots/fig03d_modularity_hESC.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `*.tsv`
- `.csv`
- `.pdf`
- `.tsv`
- `/mnt/10T/yzn/benchmark_GRN`
- `_modularity_6models_3extract.pdf`
- `_processed-network.csv`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [fig03d_modularity_hESC.py](fig03d_modularity_hESC.py)
