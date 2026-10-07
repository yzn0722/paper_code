# `tests/test_pseudotime_validation.py`

CPU tests for finite pseudotime handling and both independent dataset runners.
Requires NumPy, pandas, PyTorch, Matplotlib and scikit-learn. External model
imports, native scGPT binning and scFoundation inference are stubbed explicitly;
no real checkpoints or private datasets are used.

Tests verify numeric coercion, NaN/infinity removal, cell-order alignment,
unchanged quantile behavior for finite data, invalid schemas and duplicate IDs,
no usable cells, invalid quantiles, overlapping groups, and failure before model
inference. Both actual dataset-runner paths export identical curves and per-gene
results for dirty input versus the equivalent manually cleaned input; diagnostic
counts and scFoundation metadata are checked too.

```sh
python -m unittest discover -s tests -p test_pseudotime_validation.py -v
```

These tests validate input handling and export integration, not real pretrained
inference or regeneration of previously reported figures.
