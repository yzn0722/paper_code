# `upstream/baselines/classical/run_beeline_baselines.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

Run BEELINE classical baselines PIDC (Python) and PPCOR (R), then evaluate
AUPR / AUPR_Ratio / EPR against STRING / Non_CHIP / CHIP.

Expression is taken from STRING processed matrices (same as expression baselines).

Example:
  python -u src/GRN_inferance/classical/run_beeline_baselines.py     --methods PIDC PPCOR     --datasets hESC hHep mDC mHSC-E mHSC-GM mHSC-L     --gt-types STRING Non_CHIP CHIP     --outdir outputs/beeline_baselines

## Dependencies

Imports found in the source (standard library and external modules): `__future__`, `argparse`, `datetime`, `json`, `numpy`, `pandas`, `pathlib`, `pidc`, `sklearn`, `subprocess`, `sys`, `time`, `typing`.

## Defined interfaces

`norm_gene`, `load_gt`, `evaluate_aupr_epr`, `ppcor_shrinkage_spearman`, `parse_ppcor_outfile`, `run_ppcor`, `run_pidc`, `collect_union_genes`, `parse_args`, `main`

## Command-line parameters

- `--gt-root`
- `--expr-source`
- `--datasets`
- `--gt-types`
- `--methods`
- `--ppcor-pval`
- `--pidc-bins`
- `--pidc-discretizer`
- `--pidc-gene-mode`: all=all genes in expression; union_gt=genes appearing in any requested GT
- `--rscript`
- `--outdir`
- `--skip-existing`

## Invocation

Run `python upstream/baselines/classical/run_beeline_baselines.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `.tsv`
- `/mnt/10T/yzn/benchmark_GRN/input_process`
- `_chip_matched-ExpressionData.csv`
- `_expr_for_ppcor.csv`
- `_processed-ExpressionData.csv`
- `_wide.csv`
- `beeline_baselines_aupr_epr.csv`
- `beeline_baselines_summary_mean_std.csv`
- `run_meta.json`
- `{dataset}_chip_matched-network.csv`
- `{dataset}_processed-network.csv`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [run_beeline_baselines.py](run_beeline_baselines.py)
