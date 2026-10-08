#!/usr/bin/env python3
"""Current Fig. 5c entry; legacy BA-consumer API retained for compatibility."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fig05_scgpt_panels import main, EPS_DIR, direction_scores
DATASETS = ['hESC','hHep','mDC','mHSC-E','mHSC-GM','mHSC-L']

def scores(pred_delta: np.ndarray, true_delta: np.ndarray, idx: np.ndarray,
           eps: float = EPS_DIR) -> tuple[float, float]:
    result = direction_scores(pred_delta, true_delta, idx, eps)
    return result["ordinary_accuracy"], result["balanced_accuracy"]

def load_saved_model(name: str, root: Path, array_name: str, sign: float, ref_json: Path):
    metadata_path = root / "metric_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else None
    eps = float(metadata["eps_dir"]) if metadata else EPS_DIR
    if metadata:
        if metadata.get("metric") != "balanced_accuracy" or metadata.get("metric_version") != 1:
            raise ValueError(f"Unsupported metric metadata: {metadata_path}")
        reference_path = root / "balanced_accuracy_curves.json"
    else:
        reference_path = ref_json
        print(f"{name}: validating legacy ordinary-accuracy reference; recalculating BA with eps={eps}")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    curves: dict[str, list[float]] = {}
    for ds in DATASETS:
        ds_dir = root / "per_dataset" / ds
        frame = pd.read_csv(ds_dir / "per_gene_final_changes.csv")
        true_delta = pd.to_numeric(frame["true_delta"], errors="raise").to_numpy(float)
        idx = np.flatnonzero(frame["in_top_eval"].astype(bool).to_numpy())
        pred_by_iter = np.load(ds_dir / array_name).astype(float) * sign
        acc_curve, ba_curve = [], []
        for row in pred_by_iter:
            acc, ba = scores(row, true_delta, idx, eps)
            if metadata is None:
                # Historical files encoded exact-zero prediction/truth as Down.
                # Reproduce only for provenance validation, never for the new BA.
                acc = float(np.mean(np.where(row[idx] > 0, 1, -1)
                                    == np.where(true_delta[idx] > 0, 1, -1)))
            acc_curve.append(acc)
            ba_curve.append(ba)
        ref = np.asarray(reference[ds], dtype=float)
        got = np.asarray((ba_curve if metadata else acc_curve)[: len(ref)], dtype=float)
        if got.shape != ref.shape or not np.allclose(got, ref, atol=1e-7, rtol=0, equal_nan=True):
            raise RuntimeError(f"{name}/{ds}: saved arrays do not reproduce {reference_path}")
        finite = np.isfinite(got) & np.isfinite(ref)
        max_diff = float(np.max(np.abs(got[finite] - ref[finite]))) if finite.any() else 0.0
        if max_diff > 1e-7:
            raise RuntimeError(f"{name}/{ds}: saved arrays do not reproduce source metric (max diff={max_diff})")
        print(f"validated {name:8s} {ds:7s}: max source-metric diff={max_diff:.3g}")
        curves[ds] = ba_curve
    return curves

if __name__ == '__main__':
    main('c')
