# `plots/fig04a_edge_precision_hESC.py`

File-level notes generated from the current server source by static inspection.

## Dependencies

Imports found in the source (standard library and external modules): `argparse`, `datetime`, `fig3_palette`, `itertools`, `matplotlib`, `numpy`, `os`, `pandas`, `pathlib`.

## Defined interfaces

`calculate_method_precision`, `plot_three_methods_grn_curve`, `parse_args`

## Command-line parameters

- `--dataset`: Dataset name, e.g. hESC / hHep / mDC
- `--figsize`: Figure size W,H (default: 6,6)
- `--show-title`: Show figure title (default: off)
- `--percents`: Comma-separated percent values

## Invocation

Run `python plots/fig04a_edge_precision_hESC.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `.csv`
- `.pdf`
- `.tsv`
- `/mnt/10T/yzn/benchmark_GRN/evl_omipath/output_att500/scgpt/scgpt_`
- `/mnt/10T/yzn/benchmark_GRN/evl_omipath/output_emb500/scgpt/scgpt_`
- `/mnt/10T/yzn/benchmark_GRN/evl_omipath/output_embhidden500/scgpt/scGPT_`
- `/mnt/10T/yzn/benchmark_GRN/input_process/STRING/`
- `_processed-network.csv`
- `three_methods_grn_curve.pdf`
- `three_methods_grn_data.csv`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [fig04a_edge_precision_hESC.py](fig04a_edge_precision_hESC.py)
