"""Finite-pseudotime regression tests, including both independent dataset runners."""

from contextlib import ExitStack, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

DYNAMICS = Path(__file__).resolve().parents[1] / "upstream/dynamics"
sys.path.insert(0, str(DYNAMICS))
from pseudotime_utils import load_aligned_pseudotime, split_pseudotime_quantiles


def load_runner(filename):
    scgpt = ModuleType("scgpt")
    model = ModuleType("scgpt.model"); model.TransformerModel = torch.nn.Module
    tokenizer = ModuleType("scgpt.tokenizer")
    vocab = ModuleType("scgpt.tokenizer.gene_tokenizer"); vocab.GeneVocab = dict
    preprocess = ModuleType("scgpt.preprocess"); preprocess.binning = lambda row, n_bins: row
    foundation = ModuleType("pretrainmodels"); foundation.select_model = None
    modules = {"scgpt": scgpt, "scgpt.model": model, "scgpt.tokenizer": tokenizer,
               "scgpt.tokenizer.gene_tokenizer": vocab, "scgpt.preprocess": preprocess,
               "pretrainmodels": foundation}
    spec = importlib.util.spec_from_file_location("fixture_" + Path(filename).stem, DYNAMICS / filename)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, src, values, src_key_padding_mask):
        return {"mlm_output": values + 1}


class PseudotimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scgpt = load_runner("run_scgpt_gene_results.py")
        cls.foundation = load_runner("run_scfoundation_balanced.py")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cells = ["late", "nan", "early", "inf", "mid1", "bad", "mid2", "neginf"]
        self.expr = pd.DataFrame(
            [[3, 9999, 1, 9999, 2, 9999, 2, 9999],
             [2, 9999, 4, 9999, 3, 9999, 3, 9999],
             [3, 9999, 1, 9999, 2, 9999, 2, 9999]],
            index=["G0", "G1", "G2"], columns=self.cells)
        self.expr.to_csv(self.root / "expr.csv")
        # Deliberately shuffled pseudotime rows relative to expression columns.
        self.write_pt(["early", "bad", "mid2", "nan", "late", "neginf", "mid1", "inf"],
                      ["0", "invalid", "0.67", np.nan, "1", "-inf", "0.33", "inf"])
        self.cfg = {"expr_csv": str(self.root / "expr.csv"), "pt_csv": str(self.root / "pt.csv"),
                    "species": "human"}
        (self.root / "args.json").write_text('{"n_bins": 51}')

    def tearDown(self):
        self.tmp.cleanup()

    def write_pt(self, cells, values):
        pd.DataFrame({"cell": cells, "pt": values}).to_csv(self.root / "pt.csv", index=False)

    def test_filters_numeric_nan_and_infinities_with_expression_alignment(self):
        expr, pt, stats = load_aligned_pseudotime(self.expr, self.root / "pt.csv")
        self.assertEqual(expr.columns.tolist(), ["late", "early", "mid1", "mid2"])
        np.testing.assert_equal(pt, [1, 0, .33, .67])
        self.assertEqual(stats["invalid_matched_pseudotime_cells"], 4)
        self.assertEqual(stats["n_cells_after_pseudotime_filter"], 4)
        early, late, lo, hi = split_pseudotime_quantiles(pt, .2)
        self.assertEqual(expr.columns[early].tolist(), ["early"])
        self.assertEqual(expr.columns[late].tolist(), ["late"])
        self.assertTrue(np.isfinite([lo, hi]).all())

    def test_valid_input_preserves_original_quantile_policy(self):
        self.write_pt(self.cells, np.arange(8))
        expr, pt, stats = load_aligned_pseudotime(self.expr, self.root / "pt.csv")
        pd.testing.assert_frame_equal(expr, self.expr)
        early, late, lo, hi = split_pseudotime_quantiles(pt, .2)
        old_lo, old_hi = np.quantile(pt, [.2, .8])
        np.testing.assert_equal(early, pt <= old_lo)
        np.testing.assert_equal(late, pt >= old_hi)
        self.assertEqual(stats["invalid_matched_pseudotime_cells"], 0)

    def test_no_valid_shared_cells_fails(self):
        for cells, values, message in ((["early"], [np.nan], "no finite"),
                                      (["unknown"], [1], "no overlapping")):
            with self.subTest(message=message):
                self.write_pt(cells, values)
                with self.assertRaisesRegex(ValueError, message):
                    load_aligned_pseudotime(self.expr, self.root / "pt.csv")

    def test_ambiguous_cell_ids_and_bad_schema_fail(self):
        for cells in (["early", "early"], ["early", None]):
            self.write_pt(cells, [0, 1])
            with self.assertRaisesRegex(ValueError, "unique"):
                load_aligned_pseudotime(self.expr, self.root / "pt.csv")
        pd.DataFrame({"cell": ["early"]}).to_csv(self.root / "pt.csv", index=False)
        with self.assertRaisesRegex(ValueError, "requires"):
            load_aligned_pseudotime(self.expr, self.root / "pt.csv")

    def test_invalid_vectors_and_overlapping_groups_fail(self):
        for pt in ([], [np.nan, 1], [np.inf, 1], [[0, 1]], [1], [1, 1, 1]):
            with self.subTest(pt=pt), self.assertRaises(ValueError):
                split_pseudotime_quantiles(pt, .2)
        for q in (0, .5, 1, np.nan, np.inf):
            with self.subTest(q=q), self.assertRaises(ValueError):
                split_pseudotime_quantiles([0, 1], q)

    def run_scgpt(self, cfg, out):
        out.mkdir()
        module = self.scgpt
        with patch.multiple(module, MODEL_DIR=str(self.root), GEN_ITERS=1, TOP_PERCENT=100), redirect_stdout(io.StringIO()):
            return module.run_dataset("toy", cfg, TinyModel(),
                                      {"<pad>": 0, "<cls>": 1, "G0": 2, "G1": 3, "G2": 4},
                                      torch.device("cpu"), out)

    def run_foundation(self, cfg, out):
        out.mkdir()
        module = self.foundation
        def predict(**kwargs):
            values = kwargs["values_full_init"].numpy() + 1
            means = values[:, kwargs["eval_model_idx"]].mean(0)
            return [means], values
        with ExitStack() as stack:
            stack.enter_context(patch.multiple(module, N_ITERS=1, TOP_PERCENT=100,
                                               DEVICE=torch.device("cpu"), PLOT_NORM_CONFUSION=False))
            stack.enter_context(patch.object(module, "iterative_predict_curve", side_effect=predict))
            stack.enter_context(patch.object(module, "save_pca_data", return_value=(None, None, None)))
            stack.enter_context(patch.object(module, "plot_pca"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return module.run_one_dataset("toy", cfg, None, {"seq_len": 3},
                                          {"G0": 0, "G1": 1, "G2": 2}, out)

    def test_scgpt_runner_matches_manually_cleaned_results(self):
        self.compare_clean_run(self.run_scgpt, "toy_gene_result.csv")

    def test_foundation_runner_matches_manually_cleaned_results(self):
        self.compare_clean_run(self.run_foundation, "toy/gene_delta_compare.csv")
        meta = json.loads((self.root / "dirty/toy/meta.json").read_text())
        self.assertEqual(meta["pseudotime_filter"]["invalid_matched_pseudotime_cells"], 4)

    def compare_clean_run(self, run, result_file):
        curve, diag = run(self.cfg, self.root / "dirty")
        self.write_pt(["late", "early", "mid1", "mid2"], [1, 0, .33, .67])
        clean_curve, clean_diag = run(self.cfg, self.root / "clean")
        np.testing.assert_equal(curve, clean_curve)
        self.assertEqual(diag["n_cells"], 4)
        self.assertEqual(diag["pseudotime_filter"]["invalid_matched_pseudotime_cells"], 4)
        self.assertEqual(clean_diag["pseudotime_filter"]["invalid_matched_pseudotime_cells"], 0)
        pd.testing.assert_frame_equal(pd.read_csv(self.root / "dirty" / result_file),
                                      pd.read_csv(self.root / "clean" / result_file))

    def test_both_runners_reject_bad_groups_before_inference(self):
        for values in ([np.nan] * 8, [1] * 8):
            self.write_pt(self.cells, values)
            for runner, method, inference in ((self.scgpt, "run_dataset", "iterative_direction_accuracy"),
                                               (self.foundation, "run_one_dataset", "iterative_predict_curve")):
                with self.subTest(method=method, values=values), patch.object(runner, inference) as infer, redirect_stdout(io.StringIO()):
                    with self.assertRaises(ValueError):
                        if method == "run_dataset":
                            runner.run_dataset("toy", self.cfg, None, {}, torch.device("cpu"), self.root)
                        else:
                            runner.run_one_dataset("toy", self.cfg, None, {}, {}, self.root)
                    infer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
