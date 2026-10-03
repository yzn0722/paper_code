# `plots/fig03a_c_topology_radar_hESC.py`

File-level notes generated from the current server source by static inspection.

## Dependencies

Imports found in the source (standard library and external modules): `argparse`, `community`, `fig2_palette`, `matplotlib`, `networkx`, `numpy`, `pandas`, `pathlib`, `typing`, `warnings`.

## Defined interfaces

`load_network`, `load_real_network`, `_rename_edge_columns`, `string_gene1_only`, `string_gene_universe`, `string_undirected_edge_pairs`, `string_directed_ordered_pairs`, `string_edge_budget_n`, `filter_predicted_edges_g1_union_topn`, `filter_predicted_edges_to_string`, `preprocess_dataframe`, `calculate_metrics`, `_finite_float`, `normalize_metrics`, `normalize_metrics_zscore_similarity`, `normalize_metrics_minmax_fallback`, `plot_radar_chart`, `main`

## Command-line parameters

- `--undirected`: Use undirected graph mode for all topology metrics (consistent with undirected CCDF).
- `--norm-method`: Normalization to STRING for radar values.

## Invocation

Run `python plots/fig03a_c_topology_radar_hESC.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `.csv`
- `.tsv`
- `/mnt/10T/yzn/benchmark_GRN/evl_omipath`
- `/mnt/10T/yzn/benchmark_GRN/input_process/STRING`
- `_STRING.csv`
- `_network_topology_metrics.csv`
- `_network_topology_radar.pdf`
- `_processed-network.csv`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [fig03a_c_topology_radar_hESC.py](fig03a_c_topology_radar_hESC.py)
