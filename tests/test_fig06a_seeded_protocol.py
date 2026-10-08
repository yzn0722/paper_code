"""Regression tests for matched Fig. 6a inputs and provenance-safe reuse."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "upstream/dynamics"
spec = importlib.util.spec_from_file_location("seeded_fig6a", SOURCE / "run_random_scgpt_seeded.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Fig6aProtocolTests(unittest.TestCase):
    def test_cache_rejects_old_protocol_and_modified_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            metrics = {}
            for name in runner.DATASETS:
                csv = path / f"{name}_gene_result.csv"
                csv.write_text("delta_true,delta_pred\n1,1\n")
                metrics[name] = {"csv_sha256": runner.sha256_file(csv)}
            protocol = {"ema_alpha": .9, "source_sha256": {"code": "current"}}
            manifest = {"seed": 1, "protocol": protocol, "metrics": metrics}
            runner.write_json(path / "seed_manifest.json", manifest)
            runner.validate_cached_run(path, protocol, 1)
            with self.assertRaises(RuntimeError):
                runner.validate_cached_run(path, {"ema_alpha": .1}, 1)
            (path / f"{runner.DATASETS[0]}_gene_result.csv").write_text("changed")
            with self.assertRaises(RuntimeError):
                runner.validate_cached_run(path, protocol, 1)

    def test_binning_randomness_is_independent_of_model_seed(self):
        observations = []
        def run_dataset(name, cfg, model, vocab, device, stage):
            observations.append((name, float(np.random.random()), float(torch.rand(1))))
            pd.DataFrame({"in_eval": [1, 1], "delta_true": [1., -1.],
                          "delta_pred": [1., -1.]}).to_csv(stage / f"{name}_gene_result.csv", index=False)
            return [1.], {}
        evaluator = SimpleNamespace(DATASETS={name: {} for name in runner.DATASETS},
                                    run_dataset=run_dataset,
                                    balanced_direction_accuracy=lambda pred, true, ix: 1.)
        with tempfile.TemporaryDirectory() as folder:
            for model_seed in (1, 10):
                runner.set_seed(model_seed)
                runner.run_datasets(evaluator, None, None, torch.device("cpu"), Path(folder))
        self.assertEqual(observations[:5], observations[5:])

    def test_packed_qkv_reinitialized_and_seeds_distinct(self):
        tree = ast.parse((SOURCE / "build_random_scgpt_model.py").read_text(encoding="utf-8"))
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "init_weights_xavier_normal")
        scope = {"nn": nn}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "builder", "exec"), scope)
        values = []
        for seed in (1, 2):
            torch.manual_seed(seed)
            model = nn.MultiheadAttention(16, 2)
            before = model.in_proj_weight.detach().clone()
            scope["init_weights_xavier_normal"](model)
            self.assertFalse(torch.equal(before, model.in_proj_weight))
            self.assertTrue(torch.equal(model.in_proj_bias, torch.zeros_like(model.in_proj_bias)))
            values.append(model.in_proj_weight.detach().clone())
        self.assertFalse(torch.equal(*values))

    def test_plot_rejects_different_observed_inputs(self):
        tree = ast.parse((ROOT / "plots/fig06a_pretrained_vs_random_scgpt.py").read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ("_sha256_file", "validate_matched_conditions")]
        scope = {"json": json, "hashlib": hashlib, "pd": pd}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "plot", "exec"), scope)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            random_dir = root / "random"; random_dir.mkdir()
            pretrained_dir = root / "pretrained"; pretrained_dir.mkdir()
            seed_dir = random_dir / "seed_01"; seed_dir.mkdir()
            protocol = {"ema_alpha": .9,
                        "input_processing": "scgpt.preprocess.binning per cell on mapped genes"}
            observed = pd.DataFrame({"gene": ["A"], "gene_used": ["A"],
                                     "is_mapped": [1], "in_eval": [1],
                                     "true_early_mean": [1.], "true_late_mean": [2.], "delta_true": [1.]})
            pretrained_csv = pretrained_dir / "toy_gene_result.csv"
            random_csv = seed_dir / "toy_gene_result.csv"
            observed.to_csv(pretrained_csv, index=False)
            observed.to_csv(random_csv, index=False)
            runner.write_json(random_dir / "experiment_manifest.json", {"protocol": protocol})
            runner.write_json(seed_dir / "seed_manifest.json", {"protocol": protocol})
            runner.write_json(pretrained_dir / "seed_manifest.json", {"protocol": protocol,
                              "metrics": {"toy": {"csv_sha256": runner.sha256_file(pretrained_csv)}}})
            validate = scope["validate_matched_conditions"]
            validate(random_dir, pretrained_dir, ["toy"], {"toy": [random_csv]})
            observed.loc[0, "true_late_mean"] = 2.5
            observed.to_csv(random_csv, index=False)
            with self.assertRaises(ValueError):
                validate(random_dir, pretrained_dir, ["toy"], {"toy": [random_csv]})


if __name__ == "__main__":
    unittest.main()
