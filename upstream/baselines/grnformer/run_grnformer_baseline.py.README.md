# `upstream/baselines/grnformer/run_grnformer_baseline.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

GRNFormer (Hegde & Cheng) baseline under FBEval AUPR / AUPR_Ratio / EPR.

Purpose
-------
Direct competitor for the "transferable embedding / cross cell-type GRN" claim.
Uses the same evaluation protocol as expression_baselines.py and RegFormer
token-embedding GRN (full TF×gene candidate matrix; missing scores = -1).

This is the Bioinformatics GRN-inference GRNFormer, NOT the ACL "integrate
GRN into RNA FM" paper.

Protocols
---------
A) Transferable inference (default, fair vs embedding cosine):
   Official pretrained checkpoint → infer on each BEELINE dataset.
   Do NOT train on the target dataset's ground-truth edges.

B) Supervised / paper LOO (optional upper bound):
   Train with their pipeline on other cell types, then eval held-out.
   Report separately; do not mix with Protocol A tables.

Checklist (alignment)
---------------------
1. Same datasets: hESC hHep mESC mDC mHSC-E mHSC-GM mHSC-L
2. Same GT: STRING / Non_CHIP / CHIP under --gt-root
3. Same expression matrix source (default STRING folder) for all GT types
4. Same metrics: AUPR, AUPR_Ratio, EPR over TF×gene universe
5. Cite distinction: supervised edge model vs unsupervised emb similarity
6. Prefer Protocol A for the transferable claim comparison

Example (evaluate existing predictions):
  python -u FBEval/run_grnformer_baseline.py     --gt-root /mnt/10T/yzn/benchmark_GRN/input_process     --pred-dir outputs/grnformer/preds     --outdir outputs/grnformer_fbeval

Example (also call GRNFormer infer_grn.py):
  python -u FBEval/run_grnformer_baseline.py     --gt-root /mnt/10T/yzn/benchmark_GRN/input_process     --grnformer-root /path/to/GRNformer     --ckpt /path/to/GRNFormer.ckpt     --run-infer     --pred-dir outputs/grnformer/preds     --outdir outputs/grnformer_fbeval

## Dependencies

Imports found in the source (standard library and external modules): `__future__`, `argparse`, `datetime`, `json`, `numpy`, `pandas`, `pathlib`, `sklearn`, `subprocess`, `sys`, `typing`.

## Defined interfaces

`norm_gene`, `load_gt`, `load_expression_genes`, `evaluate_aupr_epr`, `resolve_net_path`, `resolve_expr_path`, `species_for`, `default_tf_path`, `ensure_single_column_tf`, `load_pred_scores`, `run_infer_one`, `pred_path_for`, `evaluate_one`, `parse_args`, `main`

## Command-line parameters

- `--gt-root`: Root for ground-truth network folders (STRING/Non_CHIP/CHIP/omnipath).
- `--expr-root`: Root for expression matrices (default: same as --gt-root). Use when OmniPath nets live under input_process1000 but preds used input_process STRING.
- `--datasets`
- `--gt-types`
- `--expr-source`: Expression folder (default STRING: shared matrix across GT types).
- `--tf-root`: Directory containing human-tfs.csv and mouse-tfs.csv
- `--pred-dir`: Directory for GRNFormer predicted-edge CSVs
- `--outdir`
- `--run-infer`: Call GRNFormer infer_grn.py before evaluation (Protocol A).
- `--grnformer-root`: Clone of github.com/BioinfoMachineLearning/GRNformer
- `--ckpt`: Pretrained GRNFormer checkpoint
- `--coexpression-threshold`
- `--max-subgraph-size`
- `--python-bin`
- `--protocol`: A=pretrained transfer (default); B=mark as supervised LOO (metadata only).

## Invocation

Run `python upstream/baselines/grnformer/run_grnformer_baseline.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `-tfs.csv`
- `.csv`
- `.onecol.csv`
- `/mnt/10T/yzn/Beeline-master`
- `/mnt/10T/yzn/benchmark_GRN/input_process`
- `Directory containing human-tfs.csv and mouse-tfs.csv`
- `_predicted-edges.csv`
- `_wide.csv`
- `grnformer_aupr_epr.csv`
- `grnformer_run_meta.json`
- `grnformer_summary_mean_std.csv`
- `predicted-edges.csv`
- `{dataset}_chip_matched-ExpressionData.csv`
- `{dataset}_chip_matched-network.csv`
- `{dataset}_processed-ExpressionData.csv`
- `{dataset}_processed-network.csv`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [run_grnformer_baseline.py](run_grnformer_baseline.py)
