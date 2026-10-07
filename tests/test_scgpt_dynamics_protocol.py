"""CPU protocol tests with a deterministic model and a native-binning stub."""

import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from contextlib import redirect_stdout
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream/dynamics"))
import run_multimodel_pseudotime as unified
import run_dynamic_grn_validation as grn

spec = importlib.util.spec_from_file_location("supp_umap", ROOT / "plots/supp01_scgpt_umap_six_datasets.py")
umap_plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(umap_plot)


class ConstantModel(torch.nn.Module):
    def forward(self, src, values, src_key_padding_mask):
        return {"mlm_output": torch.full_like(values, 20)}


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.model_dir = self.root / "model"; self.model_dir.mkdir()
        (self.model_dir / "args.json").write_text('{"n_bins": 7}')
        (self.model_dir / "vocab.json").write_text('{"A": 2, "B": 3}')
        (self.model_dir / "best_model.pt").write_bytes(b'toy-checkpoint')
        self.vocab = {"<pad>": 0, "<cls>": 1, "A": 2, "B": 3}
        chip = self.root / "CHIP"; chip.mkdir()
        (self.root / "toy").mkdir()
        cells = [f"c{i}" for i in range(10)]
        values = np.array([np.arange(1, 11), np.arange(11, 21), np.full(10, 99999)])
        pd.DataFrame(values, index=["a", "B", "UNKNOWN"], columns=cells).to_csv(chip / "toy_chip_matched-ExpressionData.csv")
        pd.DataFrame({"cell": cells, "pt": np.arange(10)}).to_csv(self.root / "toy/PseudoTime.csv", index=False)
        pd.DataFrame({"TF": ["a"], "target": ["B"]}).to_csv(chip / "toy_chip_matched-network.csv", index=False)
        (self.root / "input_process/CHIP").mkdir(parents=True)
        (self.root / "PseudoTime/toy").mkdir(parents=True)
        (self.root / "input_process/CHIP/toy_chip_matched-ExpressionData.csv").write_bytes((chip / "toy_chip_matched-ExpressionData.csv").read_bytes())
        (self.root / "PseudoTime/toy/PseudoTime.csv").write_bytes((self.root / "toy/PseudoTime.csv").read_bytes())
        self.calls = []
        preprocess = ModuleType("scgpt.preprocess")
        def binning(row, n_bins):
            self.calls.append((row.copy(), n_bins))
            # Values encode the requested bin count; deliberately no global scaling.
            return np.array([1, n_bins-1], dtype=float)
        preprocess.binning = binning
        self.modules = patch.dict(sys.modules, {"scgpt.preprocess": preprocess})
        self.modules.start()

    def tearDown(self):
        self.modules.stop(); self.tmp.cleanup()

    def test_umap_matches_formal_mapped_initialization(self):
        with patch.object(umap_plot, "ROOT", self.root):
            early, middle, late, genes, info = umap_plot.load_dataset("toy", self.vocab, 7)
        self.assertEqual(len(self.calls), 10)
        for row, n_bins in self.calls:
            self.assertEqual(row.shape, (2,))
            self.assertEqual(n_bins, 7)
            self.assertLess(row.max(), 99999)
        self.assertEqual(self.calls[0][0].tolist(), [1, 11])  # no log1p
        np.testing.assert_equal(early, np.tile([1, 6, 0], (2, 1)))
        self.assertEqual(info["ema_alpha"], .9)
        self.assertFalse(info["log1p"])
        self.assertEqual(info["mapped_genes"], 2)

    def test_umap_ema_and_frozen_tokens(self):
        x = np.array([[10, 4]], dtype=np.float32)
        with patch.object(umap_plot, "GEN_ITERS", 1):
            result = umap_plot.generate_per_cell(ConstantModel(), self.vocab, x,
                                                  ["A", "UNKNOWN"], torch.device("cpu"), "toy")
        np.testing.assert_allclose(result, [[11, 4]])

    def test_grn_default_and_bundle_use_native_mapped_bins(self):
        with patch.object(sys, "argv", ["runner", "--dataset", "toy", "--expr-root", str(self.root),
                                        "--pt-root", str(self.root), "--scgpt-model-dir", str(self.model_dir),
                                        "--scgpt-repo-dir", str(self.root), "--device", "cpu"]):
            args = grn.parse_args()
        self.assertEqual(args.ema_alpha, .9)
        self.assertFalse(args.scgpt_bin_log1p)
        with patch.object(unified, "load_scgpt_model", return_value=(ConstantModel(), self.vocab)):
            bundle = grn.build_scgpt_bundle(args)
        self.assertEqual(bundle.n_bins, 7)
        self.assertEqual(bundle.genes, ["A", "B", "UNKNOWN"])
        np.testing.assert_equal(bundle.values_tensor[0].numpy(), [0, 1, 6, 0])
        args.gen_iters = 1; args.print_every = 0
        states, _, final = grn.run_group(bundle, np.array([0]), args, "early")
        np.testing.assert_allclose(states[1], [2.9, 7.4, 0], atol=1e-6)
        self.assertEqual(final[0, 0].item(), 0)  # CLS frozen

    def test_prediction_cache_requires_current_protocol_and_input(self):
        x = np.array([[1, 6]], dtype=np.float32)
        state = {"model": ConstantModel(), "vocab": self.vocab, "n_bins": 7}
        out = self.root / "new_output"
        with patch.object(umap_plot, "MODEL_DIR", self.model_dir), patch.object(umap_plot, "OUTDIR", out), patch.object(umap_plot, "GEN_ITERS", 1), patch.object(umap_plot, "generate_per_cell", wraps=umap_plot.generate_per_cell) as generate:
            umap_plot.get_prediction("mHSC-L", x, ["A", "B"], torch.device("cpu"), state)
            umap_plot.get_prediction("mHSC-L", x, ["A", "B"], torch.device("cpu"), state)
            self.assertEqual(generate.call_count, 1)
            umap_plot.get_prediction("mHSC-L", x+1, ["A", "B"], torch.device("cpu"), state)
            self.assertEqual(generate.call_count, 2)
            with patch.object(umap_plot, "EMA_ALPHA", .8):
                umap_plot.get_prediction("mHSC-L", x+1, ["A", "B"], torch.device("cpu"), state)
            self.assertEqual(generate.call_count, 3)

    def test_unmanifested_cache_is_recomputed(self):
        out = self.root / "out"; out.mkdir()
        np.save(out / "toy_scgpt_early_to_latelike_predicted_cells.npy", [[99, 99]])
        state = {"model": ConstantModel(), "vocab": self.vocab, "n_bins": 7}
        with patch.object(umap_plot, "MODEL_DIR", self.model_dir), patch.object(umap_plot, "OUTDIR", out), patch.object(umap_plot, "GEN_ITERS", 1):
            got = umap_plot.get_prediction("toy", np.array([[1, 6]], dtype=np.float32), ["A", "B"], torch.device("cpu"), state)
        np.testing.assert_allclose(got, [[2.9, 7.4]], atol=1e-6)

    def test_invalid_mapped_expression_rejected(self):
        for values in (np.array([[np.nan, 1]]), np.array([[-1, 1]])):
            with self.assertRaises(ValueError):
                umap_plot.bin_expr_to_0_50(values, 7)

    def test_grn_execute_exports_current_protocol(self):
        with patch.object(sys, "argv", ["runner", "--dataset", "toy", "--expr-root", str(self.root),
                                        "--pt-root", str(self.root), "--scgpt-model-dir", str(self.model_dir),
                                        "--scgpt-repo-dir", str(self.root), "--device", "cpu",
                                        "--outdir", str(self.root / "out"), "--gen-iters", "2",
                                        "--n-rewired", "1", "--n-tf-probes", "0", "--print-every", "0"]):
            args = grn.parse_args()
        with patch.object(unified, "load_scgpt_model", return_value=(ConstantModel(), self.vocab)), redirect_stdout(io.StringIO()):
            grn.execute(args, {"fixture": True})
        out = self.root / "out/toy"
        states = np.load(out / "early_mean_trajectory.npy")
        np.testing.assert_allclose(states[1], [2.9, 7.4, 0], atol=1e-6)
        report = json.loads((out / "dynamic_grn_validation.json").read_text())
        self.assertEqual(report["protocol"]["ema_alpha"], .9)
        self.assertEqual(report["protocol"]["n_bins"], 7)
        self.assertFalse(report["protocol"]["log1p"])


if __name__ == "__main__":
    unittest.main()
