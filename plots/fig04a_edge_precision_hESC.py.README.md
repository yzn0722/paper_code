# `plots/fig04a_edge_precision_hESC.py`

File-level notes generated from the current server source by static inspection.

## Dependencies

Imports found in the source (standard library and external modules): `argparse`, `datetime`, `fig3_palette`, `itertools`, `matplotlib`, `numpy`, `os`, `pandas`, `pathlib`.

## Defined interfaces

`reference_edges_and_candidates`, `calculate_method_precision`, `plot_three_methods_grn_curve`, `parse_args`

## Random reference and candidate space

The precision curves and random reference use the same cleaned ground-truth gene
set. With `TFEdges=True`, candidates are ground-truth TF sources paired with all
reference genes, excluding self-links. With `TFEdges=False`, candidates are all
ordered pairs of reference genes except self-links. Duplicate reference and
prediction pairs are counted once; invalid sources/targets are excluded.

The random precision is `unique_ground_truth_edges / candidate_pairs`, not the
density among all gene pairs when TF filtering is enabled. Exported curve CSVs
include `CandidateEdgesTotal` and `RandomPrecision` so the reference can be audited.

Regression checks: `python -m unittest discover -s tests -p test_fig04a_candidates.py`.

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
