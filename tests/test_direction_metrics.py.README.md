# Dynamics balanced-accuracy regression tests

Run `python -m unittest discover -s tests -p test_direction_metrics.py -v` from
the repository root. Requires NumPy, pandas, PyTorch, SciPy and the normal plot
palette dependencies; no server datasets or pretrained checkpoints are used.

Ten tests cover imbalanced truth (ordinary accuracy 0.75 versus BA 0.50), zero
prediction at positive/zero threshold, threshold boundaries, single/empty truth
classes, nonfinite input, invalid thresholds/shapes, both runner APIs and CLI
defaults, actual scFoundation iteration/batch aggregation and CSV export, scGPT
iteration/export plus new/legacy figure reference validation, and scPRINT
iteration plus main-function CSV/JSON export. Tiny deterministic models replace
checkpoints. The scGPT fixture bypasses binning to isolate scoring; the scPRINT
fixture substitutes a minimal AnnData container and disables normalization/log
preprocessing. These are CPU integration checks, not full pretrained analyses.
