# `upstream/dynamics/run_random_scgpt_seeded.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

Re-run the Fig. 6a random scGPT control with recorded, distinct seeds.

Adapted from scGRN-Bench/FBplot/fig5/run_random_scgpt_seeded.py.
Evaluation uses the same iterative settings as pretrained scGPT
(run_scgpt_gene_results.py): top 30%, 16 iters, EMA α=0.9, five datasets.

## Dependencies

Imports found in the source (standard library and external modules): `argparse`, `datetime`, `gc`, `hashlib`, `importlib`, `json`, `numpy`, `pandas`, `pathlib`, `random`, `sys`, `torch`.

## Defined interfaces

`load_module`, `sha256_file`, `hash_model`, `set_seed`, `write_json`, `calculate_metrics`, `main`

## Command-line parameters

- `--seeds`
- `--outdir`

## Invocation

Run `python upstream/dynamics/run_random_scgpt_seeded.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `_gene_result.csv`
- `experiment_manifest.json`
- `seed_manifest.json`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [run_random_scgpt_seeded.py](run_random_scgpt_seeded.py)
