"""Finite, aligned pseudotime input and validated early/late quantile splits."""

import numpy as np
import pandas as pd


def load_aligned_pseudotime(expr: pd.DataFrame, path):
    """Read a headered cell/pseudotime CSV; discard invalid matched cells together."""
    # Cell IDs are labels: preserve literal IDs such as "nan" or "NA".
    # Missing-value coercion is applied only to the pseudotime column below.
    frame = pd.read_csv(path, keep_default_na=False)
    if frame.shape[1] < 2:
        raise ValueError(f"{path}: pseudotime CSV requires cell and pseudotime columns")
    frame = frame.iloc[:, :2].copy()
    frame.columns = ["cell", "pt"]
    if (frame["cell"].isna().any() or frame["cell"].eq("").any()
            or frame["cell"].duplicated().any()):
        raise ValueError(f"{path}: pseudotime cell IDs must be nonmissing and unique")
    frame = frame.set_index("cell")
    numeric = pd.to_numeric(frame["pt"], errors="coerce")
    common = expr.columns.intersection(frame.index)
    if not len(common):
        raise ValueError(f"{path}: no overlapping expression/pseudotime cells")
    pt = numeric.loc[common].to_numpy(dtype=float)
    valid = np.isfinite(pt)
    stats = {
        "pseudotime_rows": int(len(frame)),
        "invalid_pseudotime_rows": int((~np.isfinite(numeric.to_numpy(dtype=float))).sum()),
        "n_cells_before_pseudotime_filter": int(len(common)),
        "invalid_matched_pseudotime_cells": int((~valid).sum()),
        "n_cells_after_pseudotime_filter": int(valid.sum()),
    }
    if not valid.any():
        raise ValueError(f"{path}: no finite pseudotime values among overlapping cells")
    return expr.loc[:, common[valid]].copy(), pt[valid], stats


def split_pseudotime_quantiles(pt, quantile):
    """Reject invalid thresholds and overlapping groups before any model inference."""
    pt = np.asarray(pt, dtype=float)
    if pt.ndim != 1 or not pt.size or not np.isfinite(pt).all():
        raise ValueError("Pseudotime split requires a nonempty finite one-dimensional vector")
    if not np.isfinite(quantile) or not 0 < quantile < 0.5:
        raise ValueError("Pseudotime quantile must be between 0 and 0.5, exclusively")
    lo, hi = np.quantile(pt, [quantile, 1 - quantile])
    early, late = pt <= lo, pt >= hi
    if not early.any() or not late.any() or np.any(early & late):
        raise ValueError("Pseudotime must define nonempty, nonoverlapping early/late groups")
    return early, late, float(lo), float(hi)
