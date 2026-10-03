# `upstream/network/export_attention_tsv.py`

File-level notes generated from the current server source by static inspection.

## Dependencies

Imports found in the source (standard library and external modules): `argparse`, `pathlib`, `subprocess`, `sys`.

## Defined interfaces

`parse_args`, `resolve_script`, `main`

## Command-line parameters

- `--model`: Model name.
- `--script-dir`: Directory containing model scripts.
- `model_args`: Arguments passed through to the model script. Use '--' before model args.

## Invocation

Run `python upstream/network/export_attention_tsv.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [export_attention_tsv.py](export_attention_tsv.py)
