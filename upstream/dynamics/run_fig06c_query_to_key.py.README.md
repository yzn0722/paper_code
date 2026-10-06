# Recompute Fig. 6c with query-to-key propagation

This entry point recomputes the hESC attention, token-cosine and hidden-cosine
networks at 1k, 5k, 10k, 20k and 30k edges. It retains the saved early-cell
mean trajectory and uses 200 degree-preserving rewired networks per
representation/density. The score is the median Spearman correlation across
the first eight lags. Base seed is 20260903, with representation offsets of
0, 1009 and 2018.

```bash
python upstream/dynamics/run_fig06c_query_to_key.py \
  --trajectory-dir /path/to/hESC_trajectory \
  --benchmark-root /path/to/benchmark_GRN
```

The benchmark root supplies the CHIP expression gene order, scGPT vocabulary,
and the three network TSVs under `evl_omipath`. Only the saved trajectory is
used; this command does not rerun scGPT. Use `--expression-csv` and
`--vocab-json` to override those input paths.

Results default to `outputs/fig06c/query_to_key/`: three primary JSON reports,
commands, PDF/SVG/PNG, source-data CSV and a plotting manifest. The grey
reference pools 3 x 200 rewired scores from these same JSON reports.
Observed error bars remain sample SD across eight lags, not replicate SD.
No manuscript is modified and no data are uploaded to GitHub.
