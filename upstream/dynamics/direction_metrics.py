"""Shared direction scoring for the unified and scPRINT dynamics runners."""

import json
from pathlib import Path

import numpy as np

EPS_DIR = 1e-3


def direction_signs(delta, eps=EPS_DIR):
    """Return Up=1, Down=-1, undefined=0; nonfinite values are undefined."""
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("Direction threshold must be finite and nonnegative")
    delta = np.asarray(delta, dtype=float)
    return np.where(
        np.isfinite(delta),
        np.where(delta > eps, 1, np.where(delta < -eps, -1, 0)),
        0,
    )


def direction_scores(pred_delta, true_delta, top_idx, eps=EPS_DIR):
    """Macro recall over observed truth classes; undefined predictions are wrong.

    Select top_idx before excluding near-zero/nonfinite truth. The caller must
    rank only mapped genes. If just one truth class remains, use its recall,
    matching the formal evaluators; an empty set yields NaN. Inverted-truth BA
    uses the same eligible genes and denominators, and need not equal 1 - BA.
    """
    pred = np.asarray(pred_delta, dtype=float)
    truth = np.asarray(true_delta, dtype=float)
    if pred.ndim != 1 or truth.ndim != 1 or pred.shape != truth.shape:
        raise ValueError("Prediction and truth must be matching one-dimensional arrays")
    idx = np.asarray(top_idx, dtype=int)
    td = direction_signs(truth[idx], eps)
    pd = direction_signs(pred[idx], eps)
    recalls, inverse_recalls = [], []
    for cls in (-1, 1):
        rows = td == cls
        if rows.any():
            recalls.append(float(np.mean(pd[rows] == cls)))
            inverse_recalls.append(float(np.mean(pd[rows] == -cls)))
    valid = td != 0
    return {
        "balanced_accuracy": float(np.mean(recalls)) if recalls else float("nan"),
        "inverted_balanced_accuracy": float(np.mean(inverse_recalls)) if recalls else float("nan"),
        "ordinary_accuracy": float(np.mean(pd[valid] == td[valid])) if valid.any() else float("nan"),
        "n_scored": int(valid.sum()),
        "n_up": int((td == 1).sum()),
        "n_down": int((td == -1).sum()),
    }


def balanced_direction_accuracy(pred_delta, true_delta, top_idx, eps=EPS_DIR):
    scores = direction_scores(pred_delta, true_delta, top_idx, eps)
    return scores["balanced_accuracy"], scores["inverted_balanced_accuracy"]


def metric_metadata(eps=EPS_DIR):
    direction_signs([], eps)  # Validate even before any dataset runs.
    return {
        "metric": "balanced_accuracy",
        "metric_version": 1,
        "eps_dir": float(eps),
        "evaluation_pool": "top_percent_of_mapped_genes_by_absolute_true_delta",
        "near_zero_truth": "excluded_after_top_selection",
        "near_zero_prediction": "incorrect_for_Up_or_Down_truth",
        "nonfinite_truth": "excluded",
        "nonfinite_prediction": "incorrect_for_Up_or_Down_truth",
        "single_truth_class": "recall_of_present_class",
        "empty_truth_set": "NaN",
        "accuracy_curves.json": "compatibility alias of balanced_accuracy_curves.json",
    }


def save_accuracy_curves(outdir, curves, eps=EPS_DIR):
    """Keep the legacy filename, while making the metric explicit for consumers."""
    outdir = Path(outdir)
    text = json.dumps(curves, indent=2)
    for name in ("balanced_accuracy_curves.json", "accuracy_curves.json"):
        (outdir / name).write_text(text, encoding="utf-8")
    (outdir / "metric_metadata.json").write_text(
        json.dumps(metric_metadata(eps), indent=2), encoding="utf-8"
    )
