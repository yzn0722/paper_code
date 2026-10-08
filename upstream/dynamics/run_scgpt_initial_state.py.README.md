# `upstream/dynamics/run_scgpt_initial_state.py`

Recompute three mHSC-L starting groups using the formal scGPT evaluator. Requires the same model/data environment as `run_scgpt_gene_results.py` and a completed current pretrained run.

```bash
python upstream/dynamics/run_scgpt_initial_state.py --pretrained-dir /path/to/current/pretrained --outdir /path/to/new/initial_states --iterations 8 --max-cells 16 --selection-seed 0
```

The output directory must not already exist. Input, model, configuration, vocabulary, and reference CSV checksums are validated. Preprocessing uses the recorded dataset-specific seed, native per-cell quantile bins on mapped genes, and no log1p. Reference early/late means and the mapped top-30% pool must match the pretrained run. Each early, intermediate, and late group samples up to 16 cells with seed 0; exact selected cell IDs are recorded. The same strict formal checkpoint loader and per-cell reconstruction update (0.9 old + 0.1 prediction) are used. States remain continuous after initialization, CLS/padding are frozen, and population means are recorded after each round.

Each group's predicted change is relative to its own initial mean, scored against the fixed full early-to-late binned reference using balanced accuracy and EPS=0.001. This is an initial-state sensitivity analysis, not a biological replicate experiment. The early sampled curve can differ from the full early-population curve in Fig. 5c. Outputs include trajectories, a pickle-free reference NPZ, and a JSON manifest. Data and outputs are not included in the repository.
