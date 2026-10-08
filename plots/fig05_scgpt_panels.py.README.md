# `plots/fig05_scgpt_panels.py`

Replot Fig. 5c/d/e from explicit current scGPT results. Requires Python, NumPy, pandas, SciPy, and Matplotlib. Run from the repository root:

```bash
python plots/fig05_scgpt_panels.py --pretrained-dir /path/to/current/pretrained --curves-json /path/to/previous/curves_all_models.json --trajectory-dir /path/to/new/initial_states --outdir /path/to/new/figures
```

The pretrained directory must contain the native-binning EMA=0.9 seed manifest and dataset CSV. The CSV checksum, mapping flags, top-30% pool, and balanced accuracy must reproduce the recorded result. Panel c preserves other model curves and replaces only scGPT; this preservation does not validate other model results. In particular, historical LangCell dynamic curves produced using a random output head require separate recomputation with a trained compatible MLM head or removal from validated comparisons.

Panel c defaults to the first 11 iterations (use `--c-iterations 16` for the full run). Panel d requires the new initial-state runner output and verifies its reference against the pretrained manifest. Panel e uses all mapped genes at iteration 16, excludes OOV genes, and applies EPS=0.001 for colour labels; near-zero observed directions are shown in grey if present. Continuous scatter correlation uses all mapped genes, while BA uses the mapped top 30%. Gene-wise Pearson P is descriptive; genes are not independent biological replicates.

Exports editable PDF/SVG, 600-dpi PNG previews, source CSVs, and provenance JSON. Single deterministic trajectories have no inferential error bars. Files containing data, weights, figures, and outputs are external and excluded from this repository.
