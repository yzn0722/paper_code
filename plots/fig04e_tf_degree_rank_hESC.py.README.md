# `plots/fig04e_tf_degree_rank_hESC.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

绘制TF出度分布图：突出显示重叠TF（按Degree排序，内部读Outdegree列）

## Dependencies

Imports found in the source (standard library and external modules): `matplotlib`, `numpy`, `pandas`, `pathlib`, `warnings`.

## Defined interfaces

`plot_tf_outdegree_scatter`, `plot_log_scale_version`, `create_summary_table`, `main`

## Invocation

Run `python plots/fig04e_tf_degree_rank_hESC.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `/mnt/10T/yzn/scGRN-Bench/FBplot/fig3/tf_vene`
- `/mnt/10T/yzn/scGRN-Bench/FBplot/fig3/tf_vene/all_tfs_with_outdegree.csv`
- `overlap_tf_degree_sorted_summary.csv`
- `tf_degree_sorted_scatter.pdf`
- `tf_degree_sorted_scatter.png`
- `tf_degree_sorted_scatter_log.pdf`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [fig04e_tf_degree_rank_hESC.py](fig04e_tf_degree_rank_hESC.py)
