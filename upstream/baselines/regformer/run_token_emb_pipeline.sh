#!/usr/bin/env bash
# Prepare BEELINE h5ad + run RegFormer token-emb GRN (no GPU/mamba needed)
set -euo pipefail
ROOT=/mnt/10T/yzn/RegFormer
PY=${PY:-/mnt/10T/yzn/anconda3/envs/gears/bin/python}

"$PY" "$ROOT/scripts/prepare_beeline_h5ad.py"
"$PY" "$ROOT/scripts/run_grn_token_emb.py" \
  --ckpt-dir "$ROOT/checkpoints/extracted/RegFormer-10k" \
  --data-dir "$ROOT/data" \
  --tf-file "$ROOT/resources/hs_hgnc_tfs.txt" \
  --outdir "$ROOT/outputs/grn_token_emb" \
  --top-k 0

echo "Official Mamba GRN (needs CUDA+mamba_ssm):"
echo "  cd $ROOT && python downstream_task/regformer_grn.py --config_file Docs/configs/grn_hESC.toml"
