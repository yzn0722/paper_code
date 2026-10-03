# `plots/fig04f_tf_enrichment_hESC.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

Run ORA enrichment from 11 TF genes and plot lollipop only (no gene-concept network)
Top 15 pathways fully forced mapped for correct naming.

## Dependencies

Imports found in the source (standard library and external modules): `__future__`, `fig3_palette`, `math`, `matplotlib`, `numpy`, `pandas`, `pathlib`, `re`, `typing`.

## Defined interfaces

`read_gmt`, `hypergeom_p_over`, `bh_fdr`, `run_ora`, `simplify_pathway_name`, `process_and_deduplicate`, `plot_lollipop`, `main`

## Invocation

Run `python plots/fig04f_tf_enrichment_hESC.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `/mnt/10T/yzn/benchmark_GRN/fuji/c2.cp.v2025.1.Hs.symbols.gmt`
- `/mnt/10T/yzn/benchmark_GRN/fuji/h.all.v2025.1.Hs.symbols.gmt`
- `tf_overlap_11genes_ora.csv`
- `tf_overlap_11genes_ora_lollipop.pdf`
- `tf_vene/tf_overlap_detailed_statistics.csv`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [fig04f_tf_enrichment_hESC.py](fig04f_tf_enrichment_hESC.py)
