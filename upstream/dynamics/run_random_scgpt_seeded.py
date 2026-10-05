#!/usr/bin/env python3
"""Re-run the Fig. 6a random scGPT control with recorded, distinct seeds.

Adapted from scGRN-Bench/FBplot/fig5/run_random_scgpt_seeded.py.
Evaluation uses the same iterative settings as pretrained scGPT
(run_scgpt_gene_results.py): top 30%, 16 iters, EMA α=0.9, five datasets.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

FIG4 = Path(__file__).resolve().parent
MODEL_SOURCE = FIG4 / "build_random_scgpt_model.py"
EVAL_SOURCE = FIG4 / "run_scgpt_gene_results.py"
DEFAULT_OUTPUT = FIG4 / "results_multidataset_pseudotime_227_random_seeded_20260929"
DATASETS = ("hESC", "hHep", "mHSC-E", "mHSC-GM", "mHSC-L")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_model(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def calculate_metrics(file_path, evaluator):
    """Score a result CSV with the pretrained scGPT evaluator's exact policy."""
    df = pd.read_csv(file_path)
    required_cols = ["in_eval", "delta_true", "delta_pred"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"{file_path} is missing evaluator columns: {required_cols}")
    eval_idx = np.flatnonzero(df["in_eval"].to_numpy(dtype=int) == 1)
    return float(
        evaluator.balanced_direction_accuracy(
            df["delta_pred"].to_numpy(dtype=float),
            df["delta_true"].to_numpy(dtype=float),
            eval_idx,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Seeds must be distinct")
    if len(args.seeds) != 10 and args.seeds != [1]:
        raise ValueError("Use one pilot seed or exactly ten distinct seeds")
    outdir = args.outdir.resolve()
    if outdir == FIG4 / "results_multidataset_pseudotime_227_random":
        raise ValueError("Refusing to overwrite the original random results")
    outdir.mkdir(parents=True, exist_ok=True)

    random_source = load_module("fig6a_random_source", MODEL_SOURCE)
    evaluator = load_module("fig6a_evaluator", EVAL_SOURCE)
    if (
        evaluator.PT_QUANTILE,
        evaluator.TOP_PERCENT,
        evaluator.GEN_ITERS,
        evaluator.BATCH_SIZE,
        evaluator.EMA_ALPHA,
        evaluator.NO_LOG1P,
    ) != (0.2, 30, 16, 16, 0.9, True):
        raise RuntimeError("Evaluation parameters changed; review before rerunning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}; seeds={args.seeds}; output={outdir}", flush=True)
    source_hashes = {str(path): sha256_file(path) for path in (Path(__file__), MODEL_SOURCE, EVAL_SOURCE)}
    input_hashes = {
        name: {key: sha256_file(Path(evaluator.DATASETS[name][key])) for key in ("expr_csv", "pt_csv")}
        for name in DATASETS
    }

    for seed in args.seeds:
        done = outdir / f"seed_{seed:02d}"
        if done.exists():
            manifest = json.loads((done / "seed_manifest.json").read_text(encoding="utf-8"))
            if manifest.get("seed") != seed or any(
                not (done / f"{name}_gene_result.csv").exists() for name in DATASETS
            ):
                raise RuntimeError(f"Incomplete existing result: {done}")
            print(f"seed={seed} already complete; skipping", flush=True)
            continue
        stage = outdir / f".seed_{seed:02d}.partial"
        stage.mkdir(exist_ok=True)
        set_seed(seed)
        model, vocab = random_source.build_model(
            random_source.MODEL_DIR, device, use_pretrained=False
        )
        model_digest = hash_model(model)
        print(f"seed={seed} model_sha256={model_digest}", flush=True)
        metrics = {}
        for name in DATASETS:
            curve, diag = evaluator.run_dataset(
                name, evaluator.DATASETS[name], model, vocab, device, stage
            )
            csv_path = stage / f"{name}_gene_result.csv"
            ba = calculate_metrics(csv_path, evaluator)
            if ba is None or not np.isfinite(ba):
                raise RuntimeError(f"Invalid balanced accuracy for {name}, seed {seed}")
            metrics[name] = {
                "balanced_accuracy_top30": ba,
                "accuracy_curve": curve,
                "diagnostics": diag,
                "csv_sha256": sha256_file(csv_path),
            }
            print(f"seed={seed} dataset={name} top30_BA={ba:.8f}", flush=True)
        write_json(
            stage / "seed_manifest.json",
            {
                "seed": seed,
                "model_sha256": model_digest,
                "model_initialization": "Xavier/Glorot normal; pretrained weights not loaded",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics,
            },
        )
        stage.rename(done)
        del model, vocab
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    completed = {}
    for seed in range(1, 11):
        path = outdir / f"seed_{seed:02d}" / "seed_manifest.json"
        if path.exists():
            completed[str(seed)] = json.loads(path.read_text(encoding="utf-8"))["model_sha256"]
    write_json(
        outdir / "experiment_manifest.json",
        {
            "completed_seeds": completed,
            "all_ten_complete": len(completed) == 10 and len(set(completed.values())) == 10,
            "datasets": DATASETS,
            "evaluation": {
                "pt_quantile": 0.2,
                "top_percent": 30,
                "gen_iters": 16,
                "batch_size": 16,
                "ema_alpha": 0.9,
                "no_log1p": True,
            },
            "python": sys.version,
            "torch": torch.__version__,
            "device": str(device),
            "source_sha256": source_hashes,
            "input_sha256": input_hashes,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(
        f"complete={len(completed)}/10 unique_model_hashes={len(set(completed.values()))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
