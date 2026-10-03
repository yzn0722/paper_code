# `upstream/baselines/regformer/run_grn_token_emb.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

RegFormer GRN from pretrained token embeddings (no mamba_ssm / GPU required).

Uses encoder.embedding.weight from best_model.pt + cosine similarity,
TF→target scoring, exports Gene1/Gene2/EdgeWeight for FBEval.

NOTE: Official RegFormer GRN uses contextual gene embeddings from Mamba
forward (hidemb). This script is the emb-equivalent fallback when CUDA/mamba
is unavailable. Re-run with downstream_task/regformer_grn.py when GPU works.

## Dependencies

Imports found in the source (standard library and external modules): `__future__`, `anndata`, `argparse`, `json`, `numpy`, `pandas`, `pathlib`, `sklearn`, `torch`, `typing`.

## Defined interfaces

`norm`, `load_vocab`, `load_tfs`, `load_expr_genes`, `load_gt`, `evaluate`, `build_edges`, `main`

## Command-line parameters

- `--ckpt-dir`
- `--data-dir`
- `--gt-root`
- `--tf-file`
- `--datasets`
- `--gt-types`
- `--top-k`: 0=all TF-gene scores; >0=paper top-k
- `--min-sim`
- `--outdir`
- `--skip-existing`: Reuse existing prediction TSVs and only re-evaluate.

## Invocation

Run `python upstream/baselines/regformer/run_grn_token_emb.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `.h5ad`
- `.tsv`
- `/mnt/10T/yzn/RegFormer/checkpoints/extracted/RegFormer-10k`
- `/mnt/10T/yzn/RegFormer/data`
- `/mnt/10T/yzn/RegFormer/outputs/grn_token_emb`
- `/mnt/10T/yzn/RegFormer/resources/hs_hgnc_tfs.txt`
- `/mnt/10T/yzn/benchmark_GRN/input_process`
- `best_model.pt`
- `regformer_tokenemb_AUPR_Ratio_wide.csv`
- `regformer_tokenemb_aupr_epr.csv`
- `vocab.json`
- `{dataset}_chip_matched-network.csv`
- `{dataset}_processed-network.csv`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [run_grn_token_emb.py](run_grn_token_emb.py)
