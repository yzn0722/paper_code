# `upstream/dynamics/run_multimodel_pseudotime.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

Unified pseudotime benchmark entry for Geneformer / LangCell / scGPT / scFoundation.

方向准确率定义（各连续模型一致）：
  true_delta = late 细胞基因均值 − early 细胞基因均值；
  pred_delta = 迭代后预测（early 细胞上平均）− 同一 early 基线均值；
  在 |true_delta| 最大的 top% 基因上比较 sign(pred_delta) 与 sign(true_delta)。

Token 模型用排位变化经负号与表达变化对齐（见 direction_accuracy_top_genes 与代码注释）。

Examples:
  python run_unified_multidataset_pseudotime.py --model geneformer
  python run_unified_multidataset_pseudotime.py --model langcell
  python run_unified_multidataset_pseudotime.py --model scgpt
  python run_unified_multidataset_pseudotime.py --model scfoundation

## Dependencies

Imports found in the source (standard library and external modules): `argparse`, `json`, `matplotlib`, `numpy`, `os`, `pandas`, `pathlib`, `pickle`, `pretrainmodels`, `random`, `sccello`, `scgpt`, `sys`, `torch`, `transformers`, `typing`, `umap`, `warnings`.

## Defined interfaces

`parse_args`, `set_seed`, `direction_accuracy_top_genes`, `normalize_symbol`, `is_ensembl_id`, `mask_segment_bins`, `resolve_early_late_masks`, `gene_result_stem`, `read_pt_file`, `load_geneformer_dicts`, `load_sccello_ensembl_to_token_id`, `merge_vocab_with_sccello_ids`, `build_symbol_to_ensembl_map`, `LangCellModel`, `ScCelloMaskedLMWrapper`, `iterative_token_sampling`, `positions_dict_from_seq`, `run_token_model_dataset`, `bin_expr_to_0_50`, `run_scgpt_dataset`, `_scf_strip_prefix`, `scf_gather_data`, `scf_read_gene_index_tsv`, `scf_build_aligned_matrix`, `scf_make_masks_present_only`, `scf_build_io`, `scf_mean_match_calibration`, `load_scfoundation_model`, `_scf_iterative_predict_curve`, `run_scfoundation_dataset`, `_pca_2d`, `save_trajectory_plot_and_data`, `save_dataset_artifacts`, `load_scgpt_model`, `main`

## Command-line parameters

- `--model`
- `--outdir`
- `--datasets`: Comma-separated dataset names to run (default: run all). Example: --datasets mDC,hESC
- `--save-iterations`: Save per-iteration outputs and per-gene final changes (default: enabled).
- `--seed`
- `--save-cell-preds-by-iter`: Save per-iteration per-cell prediction states (disk-heavy). Token models save token-id states; continuous models save full predicted expression/state matrices.
- `--device`: Compute device. auto: prefer CUDA if available.
- `--pt-quantile`
- `--pt-mask`: quantile: early/late = bottom/top pt_quantile (default). segment_pair: early=--segment-early bin, late=--segment-late bin (equal-frequency PT segments).
- `--n-pt-segments`: Number of equal-frequency PT bins when --pt-mask=segment_pair.
- `--segment-early`: 0-based segment index for early cells (segment_pair mode).
- `--segment-late`: 0-based segment index for late cells (segment_pair mode; typically early+1).
- `--top-percent`
- `--gen-iters`
- `--batch-size`
- `--max-early-cells`: Token models only: 参与迭代的 early 细胞数上限（0=使用全部 early，与 true_delta 的 early 定义一致）。旧版曾只用前 batch_size 个细胞，会导致预测均值与全数据上的 late−early 不对齐。
- `--acc-eps`: Balanced accuracy threshold (default `1e-3`): near-zero truth excluded; near-zero prediction incorrect; exact zero remains undefined even at eps=0.
- `--max-len`
- `--mask-ratio`
- `--topk-sample`
- `--temperature`
- `--enforce-unique`
- `--token-update`: Token models only: 'replace' (MLM sampling) or 'swap' (reorder by swapping existing tokens).
- `--swap-trials`: Token models only, swap mode: number of swap proposals per iteration per cell.
- `--force-nondecreasing-acc`: Token models only: revert iteration updates when accuracy decreases.
- `--restrict-to-dataset-genes`: For token models: only sample token_ids that correspond to mapped genes in the current dataset.
- `--use-log1p`
- `--no-log1p`
- `--geneformer-model-dir`
- `--geneformer-dicts-dir`
- `--langcell-model-dir`
- `--sccello-model-dir`
- `--sccello-repo-dir`: Contains sccello/src/*.py
- `--sccello-cuda`: Use CUDA for sccello. Default off because this model path frequently triggers CUDA index asserts in this workflow.
- `--scgpt-model-dir`
- `--scgpt-repo-dir`
- `--scgpt-bin-log1p`: scGPT only: use log1p inside 0–50 binning. Default off (matches pre_scgpt/run_multidataset_pseudotime NO_LOG1P=True).
- `--scgpt-legacy-pt`: scGPT only: read pseudotime exactly like pre_scgpt/run_multidataset_pseudotime.py (pd.read_csv + rename; no read_pt_file dropna/coerce). Use if numbers must match old script.
- `--scf-root`: scFoundation: repo root (pretrainmodels lives here).
- `--scf-ckpt`: scFoundation checkpoint path.
- `--scf-gene-index-tsv`: scFoundation gene index TSV.
- `--scf-mmf-key`: Checkpoint sub-key for weights.
- `--scf-mode`: Masking mode (see pre_scfoundation/final.py).
- `--scf-value-mask-prob`
- `--scf-zero-mask-prob`
- `--scf-update-scope`
- `--scf-eps-dir`: Threshold for up/down vs zero in direction accuracy.
- `--scf-no-refresh-encoder`: If set, reuse encoder_io from iter0 (default: refresh each iter, like final.py).
- `--scf-no-resample-mask`: If set, keep the same MAE mask every iteration.
- `--scf-no-identity-input`: If set, disable identity input pass-through (unusual; default matches final.py).
- `--scf-calibration`
- `--scf-calibrate-on`
- `--ema-alpha`: EMA retention on the previous state (paper α): x <- alpha*x + (1-alpha)*x_hat. Default 0.9 => 0.1 update rate.
- `--print-every`: Print iteration progress every N iterations.
- `--save-trajectory-plot`: Save trajectory PNG and raw plotting data for each dataset.
- `--trajectory-embed`: Embedding method for trajectory plot.

## Invocation

Run `python upstream/dynamics/run_multimodel_pseudotime.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

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
- `/mnt/10T/yzn/benchmark_GRN/model/weights/Geneformer/default/6L`
- `/mnt/10T/yzn/benchmark_GRN/model/weights/Geneformer/dicts`
- `/mnt/10T/yzn/benchmark_GRN/model/weights/LangCell/cell_bert`
- `/mnt/10T/yzn/benchmark_GRN/model/weights/scCello`
- `/mnt/10T/yzn/benchmark_GRN/pre_scgpt/scGPT`
- `/mnt/10T/yzn/benchmark_GRN/pre_scgpt/scGPT/scgpt_human`
- `/mnt/10T/yzn/benchmark_GRN/sc_foundation_evals`
- `/mnt/10T/yzn/scFoundation-main/model`
- `/mnt/10T/yzn/scFoundation-main/model/OS_scRNA_gene_index.19264.tsv`
- `/mnt/10T/yzn/scFoundation-main/model/models/models.ckpt`
- `2d.csv`
- `2d.png`
- `_gene_result.csv`
- `_rank_before_after_final.csv`
- `_rank_before_after_final_top.csv`
- `accuracy_curves.json`
- `args.json`
- `best_model.pt`
- `diagnostics.json`
- `errors.json`
- `example_cell_rank_changes.csv`
- `example_seqs_by_iter.npy`
- `gene_name_id_dict.pkl`
- `mean_rank_delta_by_iter.npy`
- `per_gene_final_changes.csv`
- `pred_delta_by_iter.npy`
- `preds_full_by_iter.npy`
- `token_dictionary.pkl`
- `token_ids_by_iter.npy`
- `trajectory_embed_input_matrix.npy`
- `trajectory_raw_matrix.npy`
- `vocab.json`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [run_multimodel_pseudotime.py](run_multimodel_pseudotime.py)

## scGPT pretrained checkpoint validation

`load_scgpt_model` uses the sibling [scgpt_checkpoint.py](scgpt_checkpoint.py); deploy them together. It requires complete, shape-compatible gene/value encoder, transformer and expression decoder weights for the dynamic `mlm_output` probe. Missing/incompatible required weights, unexpected dynamic-backbone keys and conflicting QKV aliases stop execution. Packed FlashAttention/PyTorch QKV names are translated without changing tensor values. Common checkpoint wrappers and a uniform `module.` prefix are supported.

The console reports missing/unexpected keys, shape mismatches and translations; `model.checkpoint_load_report` stores the report. Unused auxiliary CLS/MVC/DAB heads may remain unloaded, with explicit diagnostics. Other model loaders are unchanged. Real checkpoint validation and full analysis reruns still require the server environment and inputs.


## Balanced accuracy output (2026-10-07)

Direction scores now use the shared `direction_metrics.py`: first select the
requested top percentage by absolute true change within mapped genes, then
exclude true deltas with `|delta| <= eps`. The score is the mean of Up and Down
recall. If only one truth class remains, use that class's recall, matching the
formal evaluators; if none remains, report NaN. Near-zero or nonfinite predictions
are incorrect for nonzero truth, and nonfinite truth is excluded. No direction,
including an exact zero at eps=0, is silently assigned to Down.

The default threshold is `1e-3`. The unified runner's scFoundation branch uses
`--scf-eps-dir`; its other branches and scPRINT use `--acc-eps`. CSV sign labels
use the same configured threshold as the curves. `final_acc_inv_truth` retains
its compatibility name but now means inverted-truth balanced accuracy; it is
not necessarily `1 - BA`, because undefined predictions are wrong in both
comparisons. Token-model rollback restores both recorded BA scores.

Outputs:
- `balanced_accuracy_curves.json`: per-dataset BA curves.
- `accuracy_curves.json`: identical compatibility alias, also BA.
- `metric_metadata.json`: metric/version, threshold and scoring policy.
- `diagnostics.json`: the same metric policy and dataset diagnostics.

Old output files are not converted by changing this code. Rerun scoring or the
analysis before replacing figure source data. Fig. 5c and the supplemental
convergence loaders now validate new BA outputs using their metadata. For old
outputs they validate the historical ordinary-accuracy reference separately,
then recalculate the displayed BA with the default threshold. Fig. 5f consumes a
static summary CSV whose original producer has not been located; this code
change alone does not update or verify it. Fig. 6a recalculates BA from scGPT
per-gene CSVs and does not consume these accuracy-curve JSON files; Fig. 6c/d use
propagation scores rather than these direction scores.

Validation: both command-line `--help` paths work without loading checkpoints.
CPU regression fixtures cover the 0.75 ordinary accuracy / 0.50 BA example,
near-zero handling, single/empty classes, model iteration, CSV/JSON export and
figure loaders. These fixtures use tiny deterministic models; full pretrained
runs still require the external model dependencies, checkpoints and datasets.
