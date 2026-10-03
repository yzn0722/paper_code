#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert BEELINE STRING ExpressionData CSVs to AnnData h5ad for RegFormer."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

EXPR_ROOT = Path("/mnt/10T/yzn/benchmark_GRN/input_process/STRING")
OUT_DIR = Path("/mnt/10T/yzn/RegFormer/data")
DATASETS = ["hESC", "hHep", "mESC", "mDC", "mHSC-E", "mHSC-GM", "mHSC-L"]


def convert_one(ds: str) -> Path:
    expr_path = EXPR_ROOT / f"{ds}_processed-ExpressionData.csv"
    if not expr_path.is_file():
        raise FileNotFoundError(expr_path)
    expr = pd.read_csv(expr_path, index_col=0)
    # genes × cells → cells × genes
    X = expr.T.copy()
    X.index = X.index.astype(str)
    X.columns = X.columns.astype(str)
    adata = ad.AnnData(X.to_numpy(dtype=np.float32))
    adata.obs_names = X.index
    adata.var_names = X.columns
    adata.var_names_make_unique()
    # RegFormer infer_dataset requires a cell type column
    adata.obs["cell_type"] = "bulk"
    adata.obs["batch"] = "0"
    out = OUT_DIR / f"{ds}.h5ad"
    adata.write_h5ad(out)
    print(f"[ok] {ds}: {adata.n_obs} cells × {adata.n_vars} genes → {out}")
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        try:
            convert_one(ds)
        except Exception as exc:
            print(f"[skip] {ds}: {exc}")


if __name__ == "__main__":
    main()
