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
import inspect
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
DEFAULT_OUTPUT = FIG4.parents[1] / "outputs/fig06a_native_binning_ema09_20261008/random"
DATASETS = ("hESC", "hHep", "mHSC-E", "mHSC-GM", "mHSC-L")
PREPROCESSING_SEED = 20261008


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


def validate_cached_run(folder, protocol, seed):
    manifest = json.loads((folder / "seed_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("seed") != seed or manifest.get("protocol") != protocol:
        raise RuntimeError(f"Existing result has a different/unrecorded protocol: {folder}; use a new directory")
    for name in DATASETS:
        path = folder / f"{name}_gene_result.csv"
        expected = manifest.get("metrics", {}).get(name, {}).get("csv_sha256")
        if not path.is_file() or not expected or sha256_file(path) != expected:
            raise RuntimeError(f"Incomplete or changed existing result: {path}")
    return manifest


def run_datasets(evaluator, model, vocab, device, stage):
    metrics = {}
    for index, name in enumerate(DATASETS):
        # Native binning randomly resolves tied quantiles. Keep observed inputs
        # identical across conditions; only model initialization varies by seed.
        set_seed(PREPROCESSING_SEED + index)
        curve, diag = evaluator.run_dataset(name, evaluator.DATASETS[name], model, vocab, device, stage)
        csv_path = stage / f"{name}_gene_result.csv"
        ba = calculate_metrics(csv_path, evaluator)
        if not np.isfinite(ba):
            raise RuntimeError(f"Invalid balanced accuracy for {name}")
        metrics[name] = {"balanced_accuracy_top30": ba, "accuracy_curve": curve,
                         "diagnostics": diag, "csv_sha256": sha256_file(csv_path)}
        print(f"dataset={name} top30_BA={ba:.8f}", flush=True)
    return metrics


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
    parser.add_argument("--pretrained-outdir", type=Path,
                        help="Also evaluate one fully validated pretrained model with identical inputs")
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("Seeds must be distinct")
    if sorted(args.seeds) != list(range(1, 11)) and args.seeds != [1]:
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
    # Hash helpers too: their changes invalidate reuse of existing seed results.
    for helper in ("scgpt_checkpoint.py", "pseudotime_utils.py"):
        source_hashes[str(FIG4 / helper)] = sha256_file(FIG4 / helper)
    for dependency in (evaluator.scgpt_binning, evaluator.TransformerModel):
        dependency_path = Path(inspect.getsourcefile(dependency))
        source_hashes[str(dependency_path)] = sha256_file(dependency_path)
    model_dir = Path(evaluator.MODEL_DIR)
    protocol = {
        "version": "fig06a_native_binning_ema09_v1", "source_sha256": source_hashes,
        "input_sha256": input_hashes, "checkpoint_sha256": sha256_file(model_dir / "best_model.pt"),
        "args_sha256": sha256_file(model_dir / "args.json"),
        "vocab_sha256": sha256_file(model_dir / "vocab.json"),
        "preprocessing_seed": PREPROCESSING_SEED, "preprocessing_seed_by_dataset": {
            name: PREPROCESSING_SEED + index for index, name in enumerate(DATASETS)},
        "input_processing": "scgpt.preprocess.binning per cell on mapped genes",
        "pt_quantile": evaluator.PT_QUANTILE, "top_percent": evaluator.TOP_PERCENT,
        "gen_iters": evaluator.GEN_ITERS, "batch_size": evaluator.BATCH_SIZE,
        "ema_alpha": evaluator.EMA_ALPHA, "no_log1p": evaluator.NO_LOG1P,
        "eps_dir": evaluator.EPS_DIR,
        "python": sys.version, "torch": torch.__version__, "device": str(device),
    }
    if args.pretrained_outdir is not None:
        pretrained_dir = args.pretrained_outdir.resolve()
        if pretrained_dir == outdir:
            raise ValueError("Pretrained and random directories must differ")
        if pretrained_dir.exists():
            validate_cached_run(pretrained_dir, protocol, 0)
            print("Pretrained result already complete under this exact protocol", flush=True)
        else:
            stage = pretrained_dir.with_name(pretrained_dir.name + ".partial")
            stage.mkdir(parents=True, exist_ok=False)
            set_seed(PREPROCESSING_SEED)
            model, vocab = evaluator.build_model(evaluator.MODEL_DIR, device)
            metrics = run_datasets(evaluator, model, vocab, device, stage)
            write_json(stage / "seed_manifest.json", {
                "seed": 0, "condition": "pretrained", "model_sha256": hash_model(model),
                "protocol": protocol, "checkpoint_load_report": model.checkpoint_load_report,
                "created_utc": datetime.now(timezone.utc).isoformat(), "metrics": metrics})
            stage.rename(pretrained_dir)
            del model, vocab
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    for seed in args.seeds:
        done = outdir / f"seed_{seed:02d}"
        if done.exists():
            validate_cached_run(done, protocol, seed)
            print(f"seed={seed} already complete; skipping", flush=True)
            continue
        stage = outdir / f".seed_{seed:02d}.partial"
        stage.mkdir(exist_ok=False)
        set_seed(seed)
        model, vocab = random_source.build_model(
            random_source.MODEL_DIR, device, use_pretrained=False
        )
        model_digest = hash_model(model)
        print(f"seed={seed} model_sha256={model_digest}", flush=True)
        metrics = run_datasets(evaluator, model, vocab, device, stage)
        write_json(
            stage / "seed_manifest.json",
            {
                "seed": seed,
                "model_sha256": model_digest,
                "model_initialization": "Xavier/Glorot normal; pretrained weights not loaded",
                "protocol": protocol,
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
            "protocol": protocol,
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
