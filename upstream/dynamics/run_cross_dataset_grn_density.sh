#!/usr/bin/env bash
set -euo pipefail

repo_root="/mnt/10T/yzn/scGRN-Bench"
benchmark_root="/mnt/10T/yzn/benchmark_GRN"
python_bin="/mnt/10T/yzn/anconda3/envs/singlecell/bin/python"
model_root="$benchmark_root/pre_scgpt/scGPT"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
trajectory_root="$repo_root/outputs/dynamic_grn_cross_dataset_binned_ema09"
result_root="$repo_root/outputs/grn_density_cross_dataset_binned_ema09"
datasets=(hESC hHep mDC mHSC-E mHSC-GM mHSC-L)

mkdir -p "$trajectory_root/logs" "$result_root"

for dataset in "${datasets[@]}"; do
    result="$trajectory_root/$dataset/early_mean_trajectory.npy"
    log="$trajectory_root/logs/${dataset}.log"
    if [[ -s "$result" ]]; then
        echo "Skipping completed trajectory: $dataset"
        continue
    fi
    "$python_bin" "$script_dir/run_dynamic_grn_validation.py" \
        --dataset "$dataset" \
        --outdir "$trajectory_root" \
        --expr-root "$benchmark_root/input_process" \
        --pt-root "$benchmark_root/PseudoTime" \
        --scgpt-model-dir "$model_root/scgpt_human" \
        --scgpt-repo-dir "$model_root" \
        --device cuda \
        --execute \
        --gen-iters 9 \
        --ema-alpha 0.9 \
        --max-cells 16 \
        --batch-size 4 \
        --n-rewired 1 \
        --n-tf-probes 0 \
        --max-forward-passes 5000 \
        --memory-limit-gb 4 \
        --cpu-threads 2 \
        --seed 0 2>&1 | tee "$log"
done

"$python_bin" "$script_dir/run_cross_dataset_grn_density.py" \
    --benchmark-root "$benchmark_root" \
    --trajectory-root "$trajectory_root" \
    --outdir "$result_root"
