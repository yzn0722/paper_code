# `upstream/baselines/run_expression_baselines.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

Expression-only GRN baselines (natural baselines for embedding cosine similarity).

Baselines (gene–gene scores from the scRNA expression matrix):
  - Pearson correlation  (abs optional)
  - Spearman correlation (abs optional)
  - Mutual information   (binned / continuous-discretized)

Each score is evaluated with FBEval-compatible AUPR, AUPR_Ratio, and EPR
against STRING / Non_CHIP / CHIP for every dataset.

Expression is taken from the same GT folder as the network
(STRING/Non_CHIP/CHIP matched matrices), so gene universes stay aligned.

Example:
  python -u FBEval/expression_baselines.py     --gt-root /mnt/10T/yzn/benchmark_GRN/input_process     --datasets hESC hHep mESC mDC mHSC-E mHSC-GM mHSC-L     --gt-types STRING Non_CHIP CHIP     --outdir outputs/expression_baselines

## Dependencies

Imports found in the source (standard library and external modules): `__future__`, `argparse`, `datetime`, `json`, `numpy`, `pandas`, `pathlib`, `scipy`, `sklearn`, `typing`.

## Defined interfaces

`norm_gene`, `load_gt`, `load_expression`, `corr_matrix`, `digitize_rows`, `_mi_from_contingency`, `_mi_batch_from_contingency`, `pairwise_mi_matrix`, `mi_score_dict`, `corr_score_dict`, `evaluate_aupr_epr`, `resolve_net_path`, `resolve_expr_path`, `run_one_dataset_gt`, `parse_args`, `main`

## Command-line parameters

- `--gt-root`
- `--datasets`
- `--gt-types`: Networks to evaluate against. Expression is controlled by --expr-source.
- `--expr-source`: Folder for ExpressionData (default STRING: same matrix for all GT types).
- `--methods`: Defaults use abs(corr); signed corr usually underperforms for recovery ranking.
- `--use-abs-corr`: Force abs() on pearson/spearman even if method name has no _abs
- `--mi-bins`
- `--outdir`

## Invocation

Run `python upstream/baselines/run_expression_baselines.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `/mnt/10T/yzn/benchmark_GRN/input_process`
- `_wide.csv`
- `expression_baselines_aupr_epr.csv`
- `expression_baselines_summary_mean_std.csv`
- `run_meta.json`
- `{dataset}_chip_matched-ExpressionData.csv`
- `{dataset}_chip_matched-network.csv`
- `{dataset}_processed-ExpressionData.csv`
- `{dataset}_processed-network.csv`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [run_expression_baselines.py](run_expression_baselines.py)
