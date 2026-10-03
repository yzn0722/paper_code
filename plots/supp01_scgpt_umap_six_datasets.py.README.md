# `plots/supp01_scgpt_umap_six_datasets.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

Six dataset-specific joint UMAPs, following the mHSC-L all-cell workflow.

Each panel fits its own scaler, PCA, and UMAP to its observed early,
intermediate, late, and scGPT-predicted late-like cells. Coordinates must not
be compared across panels as a shared embedding.

## Dependencies

Imports found in the source (standard library and external modules): `__future__`, `audit_panel_alignment`, `gc`, `json`, `matplotlib`, `numpy`, `pandas`, `pathlib`, `scgpt`, `sklearn`, `sys`, `torch`, `umap`.

## Defined interfaces

`bin_expr_to_0_50`, `load_model`, `generate_per_cell`, `load_dataset`, `get_prediction`, `embed_dataset`, `plot_panel`, `plot_composite`, `main`

## Invocation

Run `python plots/supp01_scgpt_umap_six_datasets.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `.alignment.json`
- `.pdf`
- `.png`
- `/mnt/10T/yzn/benchmark_GRN`
- `PseudoTime.csv`
- `_chip_matched-ExpressionData.csv`
- `_data_summary.json`
- `_joint_umap_coordinates.npz`
- `_scgpt_early_to_latelike_predicted_cells.npy`
- `args.json`
- `best_model.pt`
- `mHSC-L_scgpt_early_trueLate_predLate_joint_umap_predicted_cells.npy`
- `vocab.json`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [supp01_scgpt_umap_six_datasets.py](supp01_scgpt_umap_six_datasets.py)
