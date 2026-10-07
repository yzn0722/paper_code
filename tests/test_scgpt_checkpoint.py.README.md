# `tests/test_scgpt_checkpoint.py`

CPU regression tests; requires PyTorch, without scGPT or external datasets. Run from the repository root:

```sh
python -m unittest discover -s tests -p test_scgpt_checkpoint.py -v
```

Small real PyTorch models and synthetic state dicts verify prediction parity, wrappers/DataParallel, both packed QKV alias directions, required encoder/decoder coverage, shape mismatches, conflicting aliases, extra transformer layers, unused-head diagnostics and checking the actual loader return. Rejected input checkpoints must leave model weights unchanged. The actual loader functions from both entry points are also executed in isolation with synthetic config/checkpoint files, stubbing other model imports, to verify that missing decoder weights stop execution.

These tests do not run a real FlashAttention kernel, server checkpoint, or manuscript analysis.
