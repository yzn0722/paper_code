"""CPU regression tests; no scGPT installation, data, or pretrained files needed."""

import ast
from collections import OrderedDict
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

DYNAMICS = Path(__file__).resolve().parents[1] / "upstream" / "dynamics"
sys.path.insert(0, str(DYNAMICS))
from scgpt_checkpoint import load_scgpt_dynamics_checkpoint


class TinyScGPT(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.Embedding(6, 4)
        self.value_encoder = nn.Linear(1, 4)
        layer = nn.TransformerEncoderLayer(4, 2, dim_feedforward=8, dropout=0,
                                          batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.decoder = nn.Linear(4, 1)
        self.cls_decoder = nn.Linear(4, 2)  # Not used by mlm_output.

    def forward(self, src, values):
        hidden = self.encoder(src) + self.value_encoder(values.unsqueeze(-1))
        return self.decoder(self.transformer_encoder(hidden)).squeeze(-1)


class TinyVocab(dict):
    @classmethod
    def from_file(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def append_token(self, token):
        self[token] = len(self)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.source = TinyScGPT().eval()
        self.destination = TinyScGPT().eval()
        self.state = OrderedDict((key, value.clone())
                                 for key, value in self.source.state_dict().items())

    def load(self, state, model=None):
        with redirect_stdout(io.StringIO()):
            return load_scgpt_dynamics_checkpoint(
                model if model is not None else self.destination,
                state, checkpoint_path="synthetic.pt")

    def assert_rejected_without_mutation(self, state):
        before = {key: value.clone() for key, value in self.destination.state_dict().items()}
        with self.assertRaises(RuntimeError):
            self.load(state)
        for key, value in self.destination.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)

    def test_complete_checkpoint_preserves_predictions(self):
        report = self.load(self.state)
        src = torch.tensor([[1, 2, 3], [2, 4, 5]])
        values = torch.tensor([[1., 2., 0.], [3., 1., 4.]])
        with torch.no_grad():
            torch.testing.assert_close(self.destination(src, values), self.source(src, values))
        self.assertEqual(report["status"], "loaded")
        self.assertEqual(report["missing_keys"], [])

    def test_wrappers_and_dataparallel_prefix(self):
        for wrapper in ("model_state_dict", "state_dict", "model"):
            with self.subTest(wrapper=wrapper):
                wrapped = {"module." + key: value for key, value in self.state.items()}
                self.assertEqual(self.load({wrapper: wrapped, "epoch": 5})["status"], "loaded")

    def test_flash_qkv_to_native_preserves_predictions(self):
        flash = {key.replace(".self_attn.in_proj_weight", ".self_attn.Wqkv.weight")
                 .replace(".self_attn.in_proj_bias", ".self_attn.Wqkv.bias"): value
                 for key, value in self.state.items()}
        report = self.load(flash)
        self.assertEqual(len(report["qkv_aliases"]), 2)
        self.assertEqual(report["unexpected_keys"], [])
        self.assertFalse(any("in_proj" in key for key in flash))  # Checkpoint is not mutated.
        with torch.no_grad():
            src, values = torch.tensor([[1, 3, 4]]), torch.tensor([[5., 2., 1.]])
            torch.testing.assert_close(self.destination(src, values), self.source(src, values))

    def test_native_qkv_to_flash(self):
        # Mimic FlashAttention's packed QKV parameter layout with real torch parameters.
        attn = self.destination.transformer_encoder.layers[0].self_attn
        attn.Wqkv = nn.Linear(4, 12)
        del attn.in_proj_weight
        del attn.in_proj_bias
        report = self.load(self.state)
        self.assertEqual(len(report["qkv_aliases"]), 2)
        torch.testing.assert_close(attn.Wqkv.weight, self.state[
            "transformer_encoder.layers.0.self_attn.in_proj_weight"])
        torch.testing.assert_close(attn.Wqkv.bias, self.state[
            "transformer_encoder.layers.0.self_attn.in_proj_bias"])

    def test_each_dynamic_group_is_required(self):
        for prefix in ("encoder.", "value_encoder.", "transformer_encoder.", "decoder."):
            with self.subTest(prefix=prefix):
                self.assert_rejected_without_mutation(
                    {key: value for key, value in self.state.items() if not key.startswith(prefix)})

    def test_incompatible_qkv_and_vocab_shapes_rejected(self):
        for key in ("encoder.weight", "transformer_encoder.layers.0.self_attn.in_proj_weight",
                    "decoder.weight"):
            with self.subTest(key=key):
                state = dict(self.state)
                state[key] = state[key][:-1]
                self.assert_rejected_without_mutation(state)

    def test_conflicting_qkv_aliases_rejected(self):
        state = dict(self.state)
        key = "transformer_encoder.layers.0.self_attn.in_proj_weight"
        state[key.replace("in_proj_weight", "Wqkv.weight")] = state[key] + 1
        self.assert_rejected_without_mutation(state)

    def test_unexpected_transformer_layer_rejected(self):
        state = dict(self.state)
        state["transformer_encoder.layers.1.linear1.weight"] = torch.zeros(8, 4)
        self.assert_rejected_without_mutation(state)

    def test_unused_heads_are_reported(self):
        state = {key: value for key, value in self.state.items()
                 if not key.startswith("cls_decoder.")}
        state["mvc_decoder.unused.weight"] = torch.zeros(2, 2)
        report = self.load(state)
        self.assertEqual(report["missing_keys"], ["cls_decoder.bias", "cls_decoder.weight"])
        self.assertEqual(report["unexpected_keys"], ["mvc_decoder.unused.weight"])
        self.assertEqual(report["critical_missing_keys"], [])

    def test_unexpected_loader_return_is_rejected(self):
        with patch.object(self.destination, "load_state_dict", return_value=SimpleNamespace(
                missing_keys=["decoder.weight"], unexpected_keys=[])):
            with self.assertRaisesRegex(RuntimeError, "load_state_dict returned"):
                self.load(self.state)

    def test_both_entrypoints_reject_missing_decoder(self):
        # Execute the actual loader functions in isolation: other models/data dependencies
        # are irrelevant to checkpoint validation and need not be installed for this test.
        scgpt = ModuleType("scgpt")
        scgpt_model = ModuleType("scgpt.model")
        scgpt_model.TransformerModel = TinyScGPT
        tokenizer = ModuleType("scgpt.tokenizer")
        gene_tokenizer = ModuleType("scgpt.tokenizer.gene_tokenizer")
        gene_tokenizer.GeneVocab = TinyVocab
        modules = {"scgpt": scgpt, "scgpt.model": scgpt_model,
                   "scgpt.tokenizer": tokenizer,
                   "scgpt.tokenizer.gene_tokenizer": gene_tokenizer}
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules):
            root = Path(directory)
            (root / "args.json").write_text(json.dumps({"embsize": 4, "nheads": 2,
                "d_hid": 8, "nlayers": 1, "pad_value": -2, "fast_transformer": False}))
            (root / "vocab.json").write_text(json.dumps({"<pad>": 0, "<cls>": 1, "<eoc>": 2}))
            for filename, function_name in (("run_scgpt_gene_results.py", "build_model"),
                                            ("run_multimodel_pseudotime.py", "load_scgpt_model")):
                with self.subTest(entrypoint=filename):
                    path = DYNAMICS / filename
                    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
                    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                                    and node.name == function_name)
                    isolated = ast.Module(body=[function], type_ignores=[])
                    namespace = {"__file__": str(path), "Path": Path, "sys": sys,
                                 "json": json, "torch": torch, "TransformerModel": TinyScGPT,
                                 "GeneVocab": TinyVocab}
                    exec(compile(isolated, str(path), "exec"), namespace)
                    argument = root if function_name == "build_model" else SimpleNamespace(
                        scgpt_repo_dir=str(root), scgpt_model_dir=str(root))
                    torch.save(self.state, root / "best_model.pt")
                    with redirect_stdout(io.StringIO()):
                        model, _ = namespace[function_name](argument, torch.device("cpu"))
                    self.assertEqual(model.checkpoint_load_report["status"], "loaded")
                    incomplete = {key: value for key, value in self.state.items()
                                  if not key.startswith("decoder.")}
                    torch.save(incomplete, root / "best_model.pt")
                    with redirect_stdout(io.StringIO()), self.assertRaisesRegex(
                            RuntimeError, "pretrained checkpoint rejected"):
                        namespace[function_name](argument, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
