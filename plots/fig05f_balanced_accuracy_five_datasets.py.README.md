# `plots/fig05f_balanced_accuracy_five_datasets.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

Figure 5f: five-dataset, six-model top-30% balanced-accuracy bar chart.

Input is the archived six-dataset summary CSV. mDC is deliberately excluded
because the manuscript's Figure 5f compares the other five datasets. This
script does not recompute model predictions or infer missing values.

## Dependencies

Imports found in the source (standard library and external modules): `__future__`, `argparse`, `csv`, `fig4_palette`, `hashlib`, `math`, `matplotlib`, `pathlib`.

## Defined interfaces

`read_balanced_accuracy`, `write_used_source_data`, `plot`, `main`

## Command-line parameters

- `--input`
- `--output`

## Invocation

Run `python plots/fig05f_balanced_accuracy_five_datasets.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `--output must end in .pdf`
- `.pdf`
- `.png`
- `_source_data.csv`
- `outputs/fig05f_balanced_accuracy_five_datasets.pdf`
- `source_data/fig05f_balanced_accuracy_all_models_top30.csv`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [fig05f_balanced_accuracy_five_datasets.py](fig05f_balanced_accuracy_five_datasets.py)
