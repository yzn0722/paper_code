# `plots/fig03e_f_scgpt_networks_hESC.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

Real-edge community visualization with STRING-based gene filtering.

Features:
1) Use REAL predicted edge tables (not simulated graphs).
2) Apply STRING-based gene filtering:
   Gene1 in STRING(Gene1) and Gene2 in STRING(Gene1 ∪ Gene2).
3) Support switches for dataset(s) and model(s).
4) Plot model-wise community comparison for each dataset.

## Dependencies

Imports found in the source (standard library and external modules): `__future__`, `argparse`, `community`, `fig2_palette`, `matplotlib`, `networkx`, `numpy`, `pandas`, `pathlib`, `typing`.

## Defined interfaces

`parse_args`, `_parse_list_arg`, `_normalize_edges`, `_weight_col`, `load_string_filter`, `resolve_pred_path`, `filter_pred_by_string`, `build_plot_graph`, `detect_communities`, `plot_scgpt_three_scatter`, `plot_one_dataset`, `main`

## Command-line parameters

- `--dataset`: One dataset or comma list, or 'all'
- `--models`: Comma list of models. Example: LangCell,scGPT,scFoundation
- `--extraction`: Prediction folder suffix: output_<extraction>
- `--list-datasets`: List datasets and exit
- `--list-models`: List supported models and exit
- `--max-plot-nodes`
- `--scgpt-three-scatter`: Plot only scGPT as 3-extraction scatter panels (emb500/att500/embhidden500)

## Invocation

Run `python plots/fig03e_f_scgpt_networks_hESC.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `.csv`
- `.pdf`
- `.tsv`
- `/mnt/10T/yzn/benchmark_GRN`
- `_community_scatter.pdf`
- `_processed-network.csv`
- `community_real_edges_stats_scgpt_three_scatter.csv`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [fig03e_f_scgpt_networks_hESC.py](fig03e_f_scgpt_networks_hESC.py)
