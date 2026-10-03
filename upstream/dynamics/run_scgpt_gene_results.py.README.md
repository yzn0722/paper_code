# `upstream/dynamics/run_scgpt_gene_results.py`

File-level notes generated from the current server source by static inspection.

## Dependencies

Imports found in the source (standard library and external modules): `json`, `matplotlib`, `numpy`, `os`, `pandas`, `pathlib`, `re`, `scgpt`, `sys`, `torch`, `warnings`.

## Defined interfaces

`bin_expr_to_0_50`, `convert_mouse_to_human_gene`, `direction_label`, `balanced_direction_accuracy`, `build_model`, `iterative_direction_accuracy`, `run_dataset`, `plot_nature_style`, `main`

## Invocation

Run `python upstream/dynamics/run_scgpt_gene_results.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

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
- `/mnt/10T/yzn/benchmark_GRN/pre_scgpt/scGPT`
- `/mnt/10T/yzn/benchmark_GRN/pre_scgpt/scGPT/scgpt_human`
- `_gene_result.csv`
- `accuracy_curves.json`
- `args.json`
- `best_model.pt`
- `diagnostics.json`
- `fig1_convergence.pdf`
- `fig1_convergence.png`
- `fig2_final_accuracy.pdf`
- `fig2_final_accuracy.png`
- `fig3_vocab_vs_accuracy.pdf`
- `fig3_vocab_vs_accuracy.png`
- `fig_combined.pdf`
- `fig_combined.png`
- `vocab.json`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [run_scgpt_gene_results.py](run_scgpt_gene_results.py)
