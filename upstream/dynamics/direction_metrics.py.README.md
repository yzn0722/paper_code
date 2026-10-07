# Shared dynamics direction metrics

`direction_metrics.py` centralizes BA for the unified and scPRINT runners and
Fig. 5c/supplemental convergence readers. It depends only on NumPy and the Python
standard library. It supports direct script execution and package imports.

`direction_scores(pred, truth, top_idx, eps=1e-3)` returns balanced accuracy,
inverted-truth BA, ordinary accuracy (a diagnostic only), and class counts.
Callers select top genes from the mapped pool first. Up is delta > eps; Down is
delta < -eps; other/nonfinite values are undefined. Undefined truth is excluded;
undefined prediction is incorrect. BA averages recall over the truth classes
present; a single class uses its own recall and no eligible truth yields NaN.
Exact zero remains undefined when eps=0. Negative/nonfinite thresholds and
mismatched input shapes raise ValueError.

`balanced_direction_accuracy` returns the BA/inverted-BA pair used by both
legacy-named runner APIs. `direction_signs` provides consistent CSV labels.
`save_accuracy_curves` saves the explicit BA JSON, identical legacy filename,
and versioned metric metadata; it does not reinterpret old experiment files.

Run regression tests from the repository root:

```sh
python -m unittest discover -s tests -p test_direction_metrics.py -v
```

Figure provenance: Fig. 5c and the supplemental convergence figures recalculate
scores from saved arrays. Fig. 5f reads a static source table, so its original
calculation remains unverified. Fig. 6a calculates BA directly from scGPT CSVs;
Fig. 6c/d use propagation metrics and do not read these curve JSON files.
