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

## Finite pseudotime validation

Keep the sibling `pseudotime_utils.py` alongside this evaluator. Before binning
or inference, the runner coerces pseudotime to numeric and jointly removes cells
with nonnumeric, NaN or infinite pseudotime from the expression matrix and time
vector. It preserves expression column order and the inclusive bottom/top 20%
quantile rule for valid input. No shared/valid cells, duplicate/missing time-file
cell IDs or overlapping early/late groups raise `ValueError`. Filter counts are
printed and saved under `pseudotime_filter` in diagnostics; `n_cells` counts the
retained valid shared cells. Historical outputs require rerunning to reflect
this correction.

## Pretrained checkpoint validation

`build_model` uses the sibling [scgpt_checkpoint.py](scgpt_checkpoint.py). Keep the helper beside the evaluator when copying it to the server. Before device transfer or evaluation, all gene/value encoder, transformer and expression decoder weights used by `mlm_output` must exist with matching shapes. Missing/incompatible required weights, unexpected dynamic-backbone keys and conflicting QKV aliases raise `RuntimeError`.

Packed FlashAttention `Wqkv.weight/bias` and PyTorch `in_proj_weight/bias` names are translated without changing tensors. The console reports missing/unexpected keys, shape mismatches and translations despite suppressed Python warnings. `model.checkpoint_load_report` retains the JSON-safe report. Unused auxiliary CLS/MVC/DAB heads may remain unloaded, with explicit diagnostics. Historical result validity still requires loading the real checkpoint and rerunning the analysis.


## Logic/protocol update (2026-10-08)

iterative_direction_accuracy optionally accepts trajectory_callback(iteration, population_mean). Each callback receives a copy after the same formal update, preserving default evaluation behavior and preventing recorder mutations from changing scores. This enables protocol-consistent initial-state sensitivity trajectories.
