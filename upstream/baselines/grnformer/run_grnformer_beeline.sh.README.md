# `upstream/baselines/grnformer/run_grnformer_beeline.sh`

File-level notes generated from the current server source by static inspection.

## Source notes

Protocol A: GRNFormer pretrained transfer on BEELINE, FBEval metrics.

Setup once:
git clone https://github.com/BioinfoMachineLearning/GRNformer.git /mnt/10T/yzn/GRNformer
cd /mnt/10T/yzn/GRNformer && bash setup.sh
# place/download official .ckpt, set CKPT below

Then:
bash scripts/run_grnformer_beeline.sh

## Invocation

Use `bash upstream/baselines/grnformer/run_grnformer_beeline.sh` after reviewing required arguments, environment variables, and external paths.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `/mnt/10T/yzn/Beeline-master}`
- `/mnt/10T/yzn/GRNformer`
- `/mnt/10T/yzn/GRNformer}`
- `/mnt/10T/yzn/benchmark_GRN/input_process}`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [run_grnformer_beeline.sh](run_grnformer_beeline.sh)
