# `plots/supp05_balanced_convergence_six_datasets.py`

File-level notes generated from the current server source by static inspection.

## Dependencies

Imports found in the source (standard library and external modules): `__future__`, `fig4_palette`, `json`, `matplotlib`, `numpy`, `pandas`, `pathlib`, `sklearn`, `sys`.

## Defined interfaces

`scores`, `load_saved_model`, `collect_curves`, `plot`, `main`

## Invocation

Run `python plots/supp05_balanced_convergence_six_datasets.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `/mnt/10T/yzn/benchmark_GRN/dyn4_results_unified/scgpt`
- `/mnt/10T/yzn/benchmark_GRN/pre_sccello_results_unified/sccello`
- `/mnt/10T/yzn/benchmark_GRN/pre_scprint_results_unified/scprint`
- `/mnt/10T/yzn/scGRN-Bench/FBplot/fig4`
- `_balanced_accuracy.pdf`
- `balanced_accuracy_curves_all_models.json`
- `geneformer_balanced_accuracy_curves.json`
- `interation/sccello_accuracy_curves.json`
- `interation/scgpt_accuracy_curves.json`
- `interation/scprint_accuracy_curves.json`
- `langcell_balanced_accuracy_curves.json`
- `mean_rank_delta_by_iter.npy`
- `per_gene_final_changes.csv`
- `pred_delta_by_iter.npy`
- `scfoundation_balanced_accuracy_curves.json`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [supp05_balanced_convergence_six_datasets.py](supp05_balanced_convergence_six_datasets.py)


## Metric input compatibility (2026-10-07)

The displayed curves use shared balanced Up/Down recall with `EPS_DIR=1e-3` by
default: near-zero truth is excluded and near-zero prediction is incorrect.
When a runner output directory contains `metric_metadata.json` with metric
`balanced_accuracy` and version 1, validate the saved arrays against that
root's `balanced_accuracy_curves.json` using its recorded threshold. Otherwise,
validate the existing ordinary-accuracy reference using its historical zero-as-
Down convention only for provenance, then calculate the displayed BA using the
new policy. Legacy validation is explicitly logged; it is not a plotted metric.
Thus rerun BA output is no longer rejected for disagreeing with an old ordinary-
accuracy JSON. Unsupported metric metadata and inconsistent reference arrays
raise errors. Updating this script does not replace previously exported figures.
