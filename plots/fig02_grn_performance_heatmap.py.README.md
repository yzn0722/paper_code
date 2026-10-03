# `plots/fig02_grn_performance_heatmap.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

Heatmap for fig1 data (画图.xlsx) using AUPR.py color scheme.

## Dependencies

Imports found in the source (standard library and external modules): `__future__`, `argparse`, `matplotlib`, `numpy`, `pandas`, `pathlib`.

## Defined interfaces

`_find_section_starts`, `_parse_column_layout`, `parse_panel`, `build_rgba`, `_clean_model_name`, `_gt_label`, `_draw_heatmap_block`, `plot_panel`, `main`

## Command-line parameters

- `--xlsx`
- `--outdir`

## Invocation

Run `python plots/fig02_grn_performance_heatmap.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `.pdf`
- `_EPR_AUPR_heatmap.png`
- `画图.xlsx`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [fig02_grn_performance_heatmap.py](fig02_grn_performance_heatmap.py)
