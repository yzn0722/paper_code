# `plots/supp02_scgpt_attention_heads_hESC.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

scGPT different attention heads topology metrics analysis - bar charts for all five metrics

## Dependencies

Imports found in the source (standard library and external modules): `argparse`, `fig2_palette`, `matplotlib`, `networkx`, `numpy`, `pandas`, `pathlib`, `traceback`, `typing`.

## Defined interfaces

`normalize_edges`, `detect_weight_col`, `load_string`, `find_all_heads`, `parse_head_number`, `load_filtered_pred`, `calc_avg_path_length`, `calc_clustering`, `calc_modularity`, `calc_avg_degree`, `calc_assortativity`, `calc_all_metrics`, `compute_metrics_for_heads`, `plot_combined_bar_charts`, `main`

## Command-line parameters

- `--dataset`: Dataset name
- `--top-edges`: Number of top edges to keep
- `--output-dir`: Output directory for plots and data

## Invocation

Run `python plots/supp02_scgpt_attention_heads_hESC.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `/mnt/10T/yzn/benchmark_GRN/input_process/STRING`
- `/mnt/10T/yzn/scGRN-Bench/FBplot/fig2/att_head`
- `_all_metrics_combined.pdf`
- `_head*.tsv`
- `_head_metrics.csv`
- `_processed-network.csv`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [supp02_scgpt_attention_heads_hESC.py](supp02_scgpt_attention_heads_hESC.py)
