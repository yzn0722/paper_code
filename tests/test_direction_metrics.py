"""CPU fixtures for BA, runner iteration/export, and figure input validation.

Tiny deterministic models replace checkpoints; the scPRINT test substitutes a
minimal AnnData container and disables preprocessing, isolating the score path.
"""

import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream" / "dynamics"))
sys.path.insert(0, str(ROOT / "plots"))
from direction_metrics import (balanced_direction_accuracy, direction_scores,
                               direction_signs, save_accuracy_curves)
import run_multimodel_pseudotime as unified
import run_scprint_pseudotime as scprint


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.genes = [f"G{i}" for i in range(4)]
        self.classes = ["cell"]

    def forward(self, **kwargs):
        if "values" in kwargs:
            return {"mlm_output": kwargs["values"] + 1}
        return kwargs["decoder_data"] + 1

    def on_predict_epoch_start(self):
        pass

    def _predict(self, genes, expr, depth, **kwargs):
        return {"expr": [expr + 1]}


class TinyAnnData:
    def __init__(self, X, obs=None, var=None):
        self.X = np.asarray(X)
        self.obs = obs if obs is not None else pd.DataFrame()
        self.var = var

    def __getitem__(self, item):
        return TinyAnnData(self.X[item])


def args_for(module, *argv):
    with patch.object(sys, "argv", ["runner", *argv]):
        return module.parse_args()


class MetricsTests(unittest.TestCase):
    def test_imbalanced_classes(self):
        scores = direction_scores(np.ones(4), np.array([1, 1, 1, -1]), np.arange(4))
        self.assertEqual(scores["ordinary_accuracy"], .75)
        self.assertEqual(scores["balanced_accuracy"], .5)
        self.assertEqual(scores["inverted_balanced_accuracy"], .5)

    def test_zero_prediction_is_wrong_even_at_zero_threshold(self):
        for eps in (0, 1e-3):
            self.assertEqual(balanced_direction_accuracy(np.zeros(2), np.array([-1, 1]), [0, 1], eps), (0, 0))

    def test_boundary_truth_excluded_and_prediction_wrong(self):
        scores = direction_scores([.001, -.001, 1, 0], [1, -1, .001, -.001], [0, 1, 2, 3])
        self.assertEqual(scores["n_scored"], 2)
        self.assertEqual(scores["balanced_accuracy"], 0)

    def test_single_class_and_empty(self):
        self.assertEqual(balanced_direction_accuracy([1, 0], [1, 1], [0, 1])[0], .5)
        self.assertTrue(np.isnan(balanced_direction_accuracy([1], [0], [0])[0]))

    def test_nonfinite(self):
        result = direction_scores([np.nan, 1, -1], [-1, np.inf, 1], [0, 1, 2])
        self.assertEqual(result["n_scored"], 2)
        self.assertEqual(result["balanced_accuracy"], 0)
        self.assertEqual(result["inverted_balanced_accuracy"], .5)

    def test_invalid_inputs(self):
        for eps in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                direction_signs([1], eps)
        with self.assertRaises(ValueError):
            direction_scores([1, 2], [1], [0])

    def test_both_wrappers_and_cli_defaults(self):
        for module, argv in ((unified, ["--model", "scgpt"]), (scprint, ["--checkpoint", "unused.ckpt"])):
            self.assertEqual(args_for(module, *argv).acc_eps, 1e-3)
            self.assertEqual(module.direction_accuracy_top_genes(np.ones(4), np.array([1, 1, 1, -1]), np.arange(4)), (.5, .5))


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # Fifth gene is deliberately out of vocabulary with the largest change.
        early = [1, 1, 1, 2, 1]
        late = [2, 2, 2, 1, 101]
        self.genes = [f"G{i}" for i in range(5)]
        pd.DataFrame(np.array([early, early, late, late]).T, index=self.genes,
                     columns=["c0", "c1", "c2", "c3"]).to_csv(self.root / "expr.csv")
        pd.DataFrame({"cell": ["c0", "c1", "c2", "c3"], "pt": [0, 0, 1, 1]}).to_csv(self.root / "pt.csv", index=False)
        self.cfg = {"expr_csv": str(self.root / "expr.csv"), "pt_csv": str(self.root / "pt.csv"), "species": "human"}

    def tearDown(self):
        self.tmp.cleanup()

    def test_scfoundation_actual_iteration_and_export(self):
        args = args_for(unified, "--model", "scfoundation", "--top-percent", "100", "--gen-iters", "2", "--batch-size", "1", "--print-every", "0")
        curves, diag, art = unified.run_scfoundation_dataset(
            "toy", self.cfg, TinyModel(), {"seq_len": 4, "pad_token_id": 0},
            {f"G{i}": i for i in range(4)}, torch.device("cpu"), args)
        self.assertEqual(curves, [.5, .5])
        self.assertEqual(diag["metric"], "balanced_accuracy")
        self.assertEqual(diag["final_acc_inv_truth"], .5)
        self.assertNotIn(4, art["top_idx"])
        unified.save_dataset_artifacts(self.root, "toy", args, diag, art)
        frame = pd.read_csv(self.root / "toy_gene_result.csv")
        self.assertEqual(frame["is_mapped"].tolist(), [1, 1, 1, 1, 0])
        self.assertEqual(frame["in_eval"].tolist(), [1, 1, 1, 1, 0])

    def test_scgpt_iteration_export_and_figure_consumers(self):
        (self.root / "args.json").write_text('{"n_bins": 51}')
        args = args_for(unified, "--model", "scgpt", "--scgpt-model-dir", str(self.root), "--top-percent", "100", "--gen-iters", "2", "--batch-size", "1", "--print-every", "0")
        vocab = {"<pad>": 0, "<cls>": 1, **{f"G{i}": i+2 for i in range(4)}}
        with patch.object(unified, "bin_expr_to_0_50", side_effect=lambda x, **kwargs: x):
            curves, diag, art = unified.run_scgpt_dataset("toy", self.cfg, TinyModel(), vocab, torch.device("cpu"), args)
        self.assertEqual(curves, [.5, .5])
        unified.save_dataset_artifacts(self.root, "toy", args, diag, art)
        save_accuracy_curves(self.root, {"toy": curves})
        self.assertEqual(json.loads((self.root / "accuracy_curves.json").read_text()), {"toy": [.5, .5]})
        for filename in ("fig05c_balanced_convergence_mHSC-L.py", "supp05_balanced_convergence_six_datasets.py"):
            spec = importlib.util.spec_from_file_location("plot_fixture", ROOT / "plots" / filename)
            plot = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(plot)
            plot.DATASETS = ["toy"]
            got = plot.load_saved_model("scGPT", self.root, "pred_delta_by_iter.npy", 1, self.root / "unused.json")
            self.assertEqual(got, {"toy": [.5, .5]})
            # Legacy .75 accuracy still validates, but the displayed score is BA.
            (self.root / "metric_metadata.json").unlink()
            (self.root / "legacy.json").write_text('{"toy": [0.75, 0.75]}')
            self.assertEqual(plot.load_saved_model("scGPT", self.root, "pred_delta_by_iter.npy", 1, self.root / "legacy.json"), got)
            save_accuracy_curves(self.root, {"toy": curves})

    def test_scprint_actual_iteration_and_main_export(self):
        args = args_for(scprint, "--checkpoint", "unused.ckpt", "--datasets", "toy", "--outdir", str(self.root / "out"), "--top-percent", "100", "--gen-iters", "2", "--batch-size", "1", "--normalize-target-sum", "0", "--no-log1p", "--no-populate-ontology", "--print-every", "0")
        args.log1p = False
        ad = ModuleType("anndata"); ad.AnnData = TinyAnnData
        sc = ModuleType("scanpy")
        with patch.dict(sys.modules, {"anndata": ad, "scanpy": sc}):
            result = scprint.run_one_dataset("toy", self.cfg, TinyModel(), torch.device("cpu"), args)
            self.assertEqual(result[0], [.5, .5])
            with patch.object(scprint, "parse_args", return_value=args), patch.object(scprint, "load_scprint_model", return_value=TinyModel()), patch.object(scprint, "DEFAULT_DATASETS", {"toy": self.cfg}), redirect_stdout(io.StringIO()):
                scprint.main()
        out = self.root / "out" / "scprint"
        self.assertEqual(json.loads((out / "accuracy_curves.json").read_text()), {"toy": [.5, .5]})
        self.assertEqual(json.loads((out / "errors.json").read_text()), {})
        frame = pd.read_csv(out / "per_dataset/toy/per_gene_final_changes.csv")
        self.assertEqual(frame["is_mapped"].tolist(), [True]*4)
        self.assertEqual(frame["correct_sign"].tolist(), [True, True, True, False])
        self.assertEqual(json.loads((out / "metric_metadata.json").read_text())["eps_dir"], .001)


if __name__ == "__main__":
    unittest.main()
