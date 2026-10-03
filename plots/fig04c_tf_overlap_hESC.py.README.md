# `plots/fig04c_tf_overlap_hESC.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

高级TF重叠分析：功能富集 + 网络中心性 + 调控重要性
修复set.intersection错误

## Dependencies

Imports found in the source (standard library and external modules): `collections`, `fig3_palette`, `matplotlib`, `matplotlib_venn`, `numpy`, `pandas`, `pathlib`, `scipy`, `seaborn`, `warnings`.

## Defined interfaces

`display_method_name`, `read_tf_file`, `partition_venn3_regions`, `load_and_analyze_tfs`, `calculate_overlap_metrics`, `analyze_regulatory_importance`, `create_comprehensive_plots`, `main`

## Invocation

Run `python plots/fig04c_tf_overlap_hESC.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `.csv`
- `tf_overlap_detailed_statistics.csv`
- `tf_overlap_venn.pdf`
- `tf_overlap_venn_union_catalog.pdf`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [fig04c_tf_overlap_hESC.py](fig04c_tf_overlap_hESC.py)
