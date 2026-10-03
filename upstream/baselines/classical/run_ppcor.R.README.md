# `upstream/baselines/classical/run_ppcor.R`

File-level notes generated from the current server source by static inspection.

## Source notes

BEELINE-compatible PPCOR runner (Spearman partial correlation).
Usage: Rscript run_ppcor.R <ExpressionData.csv> <outFile.txt> [pVal]

ExpressionData: genes × cells, CSV with gene names as first column / rownames.
Output raw table: Gene1, Gene2, corVal, pValue  (then Python parses like BEELINE).
BEELINE: pcor on cells × genes with Spearman
drop self-loops

## Invocation

Use `Rscript upstream/baselines/classical/run_ppcor.R` after reviewing required arguments, environment variables, and external paths.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `/home/yezhongni/R/library`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [run_ppcor.R](run_ppcor.R)
