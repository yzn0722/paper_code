"""Regression tests for paired-cell MI, true DeepSEM output, and LangCell heads."""

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "upstream/dynamics"))
from langcell_mlm import LangCellModel, load_langcell_mlm


def load_script(relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location("fixture_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = load_script("upstream/baselines/run_expression_baselines.py")

    def matrix(self, values, bins=2):
        with redirect_stdout(io.StringIO()):
            return self.baseline.pairwise_mi_matrix(pd.DataFrame(values), n_bins=bins)[0]

    def test_identical_and_independent_genes(self):
        values = np.array([[0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 0, 0, 0, 1, 1, 1, 1],
                           [0, 1, 0, 1, 0, 1, 0, 1]])
        result = self.matrix(values)
        self.assertAlmostEqual(result[0, 1], np.log(2))
        self.assertAlmostEqual(result[0, 2], 0)
        np.testing.assert_allclose(result, self.matrix(values[:, [0, 4, 1, 5, 2, 6, 3, 7]]))

    def test_all_pairs_match_contingency_reference_for_both_axis_shapes(self):
        rng = np.random.default_rng(7)
        for shape in [(3, 19), (11, 4)]:
            values = rng.integers(0, 8, size=shape)
            discrete = self.baseline.digitize_rows(values, n_bins=4)
            expected = np.array([[mutual_info_score(a, b) for b in discrete] for a in discrete])
            np.testing.assert_allclose(self.matrix(values, bins=4), expected, atol=1e-12)

    def test_constant_gene_has_zero_information(self):
        result = self.matrix([[1, 1, 1, 1], [0, 0, 1, 1]])
        np.testing.assert_allclose(result[0], 0, atol=1e-12)


class DeepSEMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_script("upstream/baselines/deepsem/run_deepsem_batch.py")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "GRN_inference_result.tsv"

    def test_true_direction_and_unrounded_small_negative_weights(self):
        self.path.write_text("TF\tTarget\tEdgeWeight\nA\tB\t0.00123456789\nB\tA\t-0.23456789123\nA\tA\t1\nB\tC\t0\n")
        result = self.runner.read_deepsem_network(self.path, ["A", "B", "C"])
        self.assertEqual(list(zip(result.Gene1, result.Gene2)), [("B", "A"), ("A", "B")])
        np.testing.assert_allclose(result.EdgeWeight, [-0.23456789123, 0.00123456789], rtol=1e-12)

    def test_invalid_or_missing_output_fails(self):
        with self.assertRaises(FileNotFoundError):
            self.runner.read_deepsem_network(self.path, ["A", "B"])
        for body in ["TF\tTarget\tEdgeWeight\nA\tUNKNOWN\t0.2\n",
                     "TF\tTarget\tEdgeWeight\nA\tB\tnan\n",
                     "TF\tTarget\tEdgeWeight\nA\tB\tinf\n",
                     "TF\tTarget\tEdgeWeight\nA\tB\t0.2\nA\tB\t0.3\n",
                     "TF\tTarget\tEdgeWeight\nA\tA\t1\n",
                     "cell\tscore\nA\t1\n"]:
            with self.subTest(body=body):
                self.path.write_text(body)
                with self.assertRaises(ValueError):
                    self.runner.read_deepsem_network(self.path, ["A", "B"])

    def fake_main(self, action):
        main = self.root / "fake DeepSEM main.py"
        main.write_text("import sys\nfrom pathlib import Path\n"
                        "folder = Path(sys.argv[sys.argv.index('--save_name') + 1])\n" + action)
        return main

    def run_wrapper(self, main):
        with patch.multiple(self.runner, DEEPSEM_MAIN_PATH=str(main), RESULT_DIR=str(self.root / "exports")), redirect_stdout(io.StringIO()):
            return self.runner.run_deepsem_grn(self.root / "expression.csv", self.root / "previous", "non_celltype_GRN", ["A", "B"], "toy")

    def test_subprocess_exports_actual_output_and_keeps_raw_network(self):
        main = self.fake_main("(folder / 'GRN_inference_result.tsv').write_text('TF\\tTarget\\tEdgeWeight\\nB\\tA\\t-0.000123456789\\n')\n")
        destination = self.run_wrapper(main)
        result = pd.read_csv(destination, sep="\t")
        self.assertEqual((result.Gene1[0], result.Gene2[0]), ("B", "A"))
        self.assertAlmostEqual(result.EdgeWeight[0], -0.000123456789, places=15)
        self.assertEqual(len(list(self.root.glob("previous_*/GRN_inference_result.tsv"))), 1)

    def test_failed_or_missing_new_output_cannot_reuse_previous_network(self):
        (self.root / "previous").mkdir()
        (self.root / "previous/GRN_inference_result.tsv").write_text("TF\tTarget\tEdgeWeight\nA\tB\t0.7\n")
        with self.assertRaises(FileNotFoundError):
            self.run_wrapper(self.fake_main("pass\n"))
        with self.assertRaises(RuntimeError):
            self.run_wrapper(self.fake_main("sys.exit(2)\n"))
        self.assertFalse((self.root / "exports/DeepSEM_toy.tsv").exists())


try:
    from transformers import BertConfig, BertForMaskedLM, BertModel
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False


@unittest.skipUnless(HAVE_TRANSFORMERS, "Install transformers to exercise real BERT checkpoint loading")
class LangCellTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        torch.manual_seed(17)
        self.config = BertConfig(vocab_size=13, hidden_size=8, num_hidden_layers=1,
                                 num_attention_heads=2, intermediate_size=16,
                                 hidden_dropout_prob=0, attention_probs_dropout_prob=0)

    def test_complete_mlm_roundtrip_preserves_logits_in_both_formats(self):
        source = BertForMaskedLM(self.config).eval()
        tokens = torch.tensor([[2, 3, 4]])
        attention = torch.ones_like(tokens)
        expected = source(tokens, attention_mask=attention).logits
        for safe in (False, True):
            path = self.root / str(safe)
            source.save_pretrained(path, safe_serialization=safe)
            torch.manual_seed(999)
            with redirect_stdout(io.StringIO()):
                loaded = LangCellModel(str(path)).eval()
            torch.testing.assert_close(loaded(tokens, attention), expected, rtol=0, atol=0)

    def test_encoder_only_is_rejected(self):
        BertModel(self.config, add_pooling_layer=False).save_pretrained(self.root)
        with self.assertRaisesRegex(RuntimeError, "trained.*MLM"):
            load_langcell_mlm(str(self.root))

    def test_missing_head_or_encoder_parameters_are_rejected(self):
        source = BertForMaskedLM(self.config)
        for name in ["cls.predictions.transform.dense.weight", "cls.predictions.bias",
                     "bert.encoder.layer.0.attention.self.query.weight"]:
            with self.subTest(name=name):
                state = dict(source.state_dict())
                del state[name]
                if name == "cls.predictions.bias":
                    del state["cls.predictions.decoder.bias"]
                source.save_pretrained(self.root, state_dict=state, safe_serialization=False)
                with self.assertRaises(RuntimeError):
                    load_langcell_mlm(str(self.root))

    def test_both_entry_points_use_same_validated_loader(self):
        independent = load_script("upstream/dynamics/run_langcell_balanced.py")
        unified = load_script("upstream/dynamics/run_multimodel_pseudotime.py")
        self.assertIs(independent.LangCellModel, LangCellModel)
        self.assertIs(unified.LangCellModel, LangCellModel)


if __name__ == "__main__":
    unittest.main()
