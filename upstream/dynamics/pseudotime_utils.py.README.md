# `upstream/dynamics/pseudotime_utils.py`

Shared finite-pseudotime handling for the independent scGPT and scFoundation
dataset runners. Requires NumPy and pandas.

`load_aligned_pseudotime(expr, path)` reads a headered CSV whose first two columns
are cell ID and pseudotime. Pseudotime is coerced to numeric; nonnumeric values,
NaN and positive/negative infinity are excluded together with their expression
columns. The result preserves expression column order and returns aligned
expression, finite pseudotime and filter counts. Missing/duplicate pseudotime
IDs, invalid schemas, no shared cells or no valid shared values raise `ValueError`.

`split_pseudotime_quantiles(pt, quantile)` requires finite nonempty 1D values and
`0 < quantile < 0.5`. It retains the original inclusive bottom/top quantile
policy, rejecting empty or overlapping early/late groups rather than producing
NaN means or evaluating the same cells in both groups.

No imputation, pseudotime reordering or expression normalization is performed.
The independent runners log removed-cell counts and save `pseudotime_filter`
metadata in their diagnostics (also scFoundation's per-dataset `meta.json`).
Keep this module beside both runners when copying scripts to a server.

Validation: `python -m unittest discover -s tests -p test_pseudotime_validation.py -v`.
