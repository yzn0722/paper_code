# `tests/test_scgpt_dynamics_protocol.py`

CPU regression tests for supplementary scGPT UMAP and GRN dynamics protocol
alignment. Uses temporary expression/pseudotime/checkpoint fixtures, a
constant-output PyTorch model and an explicit native-binning stub; no real
checkpoint, scGPT installation or dataset is needed.

Checks per-cell binning of mapped genes only, no log1p, checkpoint bin counts,
EMA retention 0.9, frozen padding/CLS values, valid cache reuse and invalidation,
recalculation of unmanifested old caches, invalid expression rejection, and the
GRN execute path's trajectory/JSON exports.

```sh
python -m unittest discover -s tests -p test_scgpt_dynamics_protocol.py -v
```

Requires NumPy, pandas, SciPy, PyTorch, Matplotlib and scikit-learn. These tests
validate the protocol's wiring and arithmetic, not native scGPT binning internals,
real pretrained model results or the completed UMAP figures.
