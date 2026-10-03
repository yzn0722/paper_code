# `plots/fig04b_tf_f1_hESC.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

TF-level Top-N overlap (ranking consistency) - Raincloud plot only.

## Dependencies

Imports found in the source (standard library and external modules): `__future__`, `argparse`, `fig3_palette`, `matplotlib`, `numpy`, `pandas`, `pathlib`, `seaborn`, `typing`, `warnings`.

## Defined interfaces

`display_method_name`, `display_method_color`, `method_sort_key_custom`, `norm_gene`, `validate_columns`, `clean_gene_df`, `load_gt_network`, `filter_prediction`, `compute_tf_metrics`, `parse_args`, `create_simple_raincloud_plot`, `main`

## Command-line parameters

- `--dataset`
- `--top-n`: Take Top-N TFs by TF-level f1
- `--min-edges-per-tf`: Only consider TFs with >= this many predicted edges
- `--score`: How to rank TFs before taking Top-N. f1_gt uses GT targets (real F1). f1_pred matches legacy f1==precision.
- `--include-genie3`: Include GENIE3 as an extra method in the overlap heatmap.
- `--gt-dir`
- `--emb500`
- `--att500`
- `--embhidden500`
- `--genie3`
- `--out`: Output directory path

## Invocation

Run `python plots/fig04b_tf_f1_hESC.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `.csv`
- `.pdf`
- `.tsv`
- `/mnt/10T/yzn/benchmark_GRN/evl_omipath/output_att500/scgpt`
- `/mnt/10T/yzn/benchmark_GRN/evl_omipath/output_emb500/GENIE3`
- `/mnt/10T/yzn/benchmark_GRN/evl_omipath/output_emb500/scgpt`
- `/mnt/10T/yzn/benchmark_GRN/evl_omipath/output_embhidden500/scgpt`
- `/mnt/10T/yzn/benchmark_GRN/input_process/STRING`
- `_processed-network.csv`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [fig04b_tf_f1_hESC.py](fig04b_tf_f1_hESC.py)
