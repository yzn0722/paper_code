# `plots/supp01_scgpt_umap_six_datasets.py`

Produces six dataset-specific joint UMAP panels of observed early, intermediate,
late, and scGPT-refined early cells. Each panel fits its own scaler, PCA and UMAP;
coordinates must not be compared between panels as a shared embedding.

## Dynamics protocol

- Match uppercased gene names to the checkpoint vocabulary.
- Apply native `scgpt.preprocess.binning` separately to each cell's mapped genes,
  using checkpoint `n_bins` (51 if absent), with log1p disabled. Unmapped genes
  are zero-filled and assigned padding tokens.
- Bin observed early/intermediate/late profiles once. Refine each early cell for
  16 iterations using `0.9 * previous + 0.1 * mlm_output`, without re-binning.
  Padding positions and the CLS value remain fixed.
- Reuse the formal dynamics loader and its critical checkpoint coverage checks.
- Reset seed 42 before each dataset so quantile tie handling does not depend on
  whether previous datasets used cached predictions.

## Inputs, dependencies and invocation

Configure `ROOT`, `SCGPT_REPO` and `MODEL_DIR` in the script for your installation.
Inputs are expression CSVs under `ROOT/input_process/CHIP`, pseudotime CSVs under
`ROOT/PseudoTime/<dataset>`, and checkpoint `args.json`, `vocab.json`, `best_model.pt`.

Requires NumPy, pandas, PyTorch, Matplotlib, scikit-learn, umap-learn and the scGPT
source/dependencies, plus this repository's dynamics modules. The main entry
requires a CUDA device. Run from the repository root:

```sh
python plots/supp01_scgpt_umap_six_datasets.py
```

## Outputs and cache migration

Outputs now use `six_dataset_joint_umap_binned_ema09` under `SOURCE_DIR`.
Older percentile-scaled/EMA-0.1 outputs, including the special old mHSC-L cache,
are not read. A prediction NPY is reusable only when its JSON manifest matches
parameters, binned input hash, gene order, checkpoint/configuration hashes and
script hash, and the prediction file hash matches. Otherwise it is regenerated.

Outputs include predicted-cell NPYs and manifests, joint coordinate NPZs,
data-summary JSONs and a six-panel PDF/PNG. The internal alignment check writes
`.alignment.json` and checks equal axes widths/heights and row/column alignment
with a 1.5-point tolerance; no external `audit_panel_alignment` module is needed.

CPU protocol validation uses deterministic fixtures in
`tests/test_scgpt_dynamics_protocol.py`; it does not establish that real pretrained
inference, UMAP embedding or manuscript figures have been rerun.

Source: [supp01_scgpt_umap_six_datasets.py](supp01_scgpt_umap_six_datasets.py)
