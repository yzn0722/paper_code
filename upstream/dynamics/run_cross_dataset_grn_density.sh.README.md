# `upstream/dynamics/run_cross_dataset_grn_density.sh`

Runs the repository's GRN trajectory generator for all six datasets, including
hESC, then computes cross-dataset GRN density statistics.

Configure `repo_root`, `benchmark_root` and `python_bin` for your installation.
The script locates both Python entry points relative to itself, avoiding an
external copy in `FBplot/fig4`. It explicitly supplies `--ema-alpha 0.9` (90% old
state, 10% prediction), leaves log1p disabled and uses the generator's native
per-cell binning of vocabulary-mapped genes.

```sh
bash upstream/dynamics/run_cross_dataset_grn_density.sh
```

Trajectory generation uses CUDA, nine iterations, at most 16 cells per group,
batch size 4 and seed 0. The wrapper writes to new folders:

- `outputs/dynamic_grn_cross_dataset_binned_ema09`
- `outputs/grn_density_cross_dataset_binned_ema09`

This prevents automatic reuse of historical EMA-0.1 trajectories. A nonempty
trajectory already present in the new folder is skipped; clear or choose a new
output folder if inputs or other settings change. Logs are saved under the
trajectory folder. Datasets, model checkpoints and generated results are not
included. Real server execution is required to obtain updated density results.

Source: [run_cross_dataset_grn_density.sh](run_cross_dataset_grn_density.sh)
