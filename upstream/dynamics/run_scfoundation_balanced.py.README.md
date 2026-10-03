# `upstream/dynamics/run_scfoundation_balanced.py`

File-level notes generated from the current server source by static inspection.

## Dependencies

Imports found in the source (standard library and external modules): `json`, `matplotlib`, `numpy`, `os`, `pandas`, `pathlib`, `pretrainmodels`, `random`, `sklearn`, `sys`, `torch`, `warnings`.

## Defined interfaces

`seed_all`, `convert_mouse_to_human_gene`, `_strip_prefix`, `gatherData`, `load_model_mmf_gene`, `read_gene_index_tsv`, `build_aligned_matrix`, `make_masks_present_only`, `build_io`, `mean_match_calibration`, `iterative_predict_curve`, `plot_norm_confusion`, `save_pca_data`, `plot_pca`, `run_one_dataset`, `plot_curves`, `main`

## Invocation

Run `python upstream/dynamics/run_scfoundation_balanced.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `.npy`
- `/mnt/10T/yzn/benchmark_GRN/PseudoTime/hESC/PseudoTime.csv`
- `/mnt/10T/yzn/benchmark_GRN/PseudoTime/hHep/PseudoTime.csv`
- `/mnt/10T/yzn/benchmark_GRN/PseudoTime/mDC/PseudoTime.csv`
- `/mnt/10T/yzn/benchmark_GRN/PseudoTime/mHSC-E/PseudoTime.csv`
- `/mnt/10T/yzn/benchmark_GRN/PseudoTime/mHSC-GM/PseudoTime.csv`
- `/mnt/10T/yzn/benchmark_GRN/PseudoTime/mHSC-L/PseudoTime.csv`
- `/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/hESC_chip_matched-ExpressionData.csv`
- `/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/hHep_chip_matched-ExpressionData.csv`
- `/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/mDC_chip_matched-ExpressionData.csv`
- `/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/mHSC-E_chip_matched-ExpressionData.csv`
- `/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/mHSC-GM_chip_matched-ExpressionData.csv`
- `/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/mHSC-L_chip_matched-ExpressionData.csv`
- `/mnt/10T/yzn/scFoundation-main/model`
- `/mnt/10T/yzn/scFoundation-main/model/OS_scRNA_gene_index.19264.tsv`
- `/mnt/10T/yzn/scFoundation-main/model/models/models.ckpt`
- `_pca.pdf`
- `_pca.png`
- `_pca_expression.npy`
- `_pca_genes.csv`
- `_pca_labels.npy`
- `acc_curve.npy`
- `accuracy_curves.json`
- `convergence_curves.png`
- `diagnostics.json`
- `gene_delta_compare.csv`
- `gene_mapping_to_model.csv`
- `meta.json`
- `norm_confusion_matrix.png`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [run_scfoundation_balanced.py](run_scfoundation_balanced.py)
