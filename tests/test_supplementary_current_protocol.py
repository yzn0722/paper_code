"""Reject stale/mismatched scGPT inputs before replacing supplementary curves."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'plots'))
import supp05_balanced_convergence_six_datasets as plot


class SupplementaryProtocolTests(unittest.TestCase):
    def fixture(self, root):
        pretrained = root / 'pretrained'
        supplemental = root / 'mdc'
        pretrained.mkdir()
        supplemental.mkdir()
        protocol = {'version': 'fixture', 'input_processing': 'scgpt.preprocess.binning per cell on mapped genes',
                    'pt_quantile': .2, 'top_percent': 30, 'gen_iters': 16, 'ema_alpha': .9,
                    'no_log1p': True, 'eps_dir': .001, 'checkpoint_sha256': 'checkpoint',
                    'args_sha256': 'args', 'vocab_sha256': 'vocab'}
        manifests = {p: {'condition': 'pretrained', 'model_sha256': 'same-model',
                         'protocol': copy.deepcopy(protocol), 'metrics': {}}
                     for p in [pretrained, supplemental]}
        current = [.5] * 15 + [1.]
        for dataset in plot.DATASETS:
            folder = supplemental if dataset == 'mDC' else pretrained
            path = folder / (dataset + '_gene_result.csv')
            pd.DataFrame({'gene': ['A', 'B', 'C', 'D', 'OOV'],
                          'is_mapped': [1, 1, 1, 1, 0], 'in_eval': [1, 0, 0, 0, 0],
                          'delta_true': [4, 3, -2, -1, 100],
                          'delta_pred': [1, 1, -1, 1, 0]}).to_csv(path, index=False)
            manifests[folder]['metrics'][dataset] = {'accuracy_curve': current,
                'balanced_accuracy_top30': 1., 'csv_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        for folder, manifest in manifests.items():
            (folder / 'seed_manifest.json').write_text(json.dumps(manifest))
        original = {model: {dataset: [.25] * 16 for dataset in plot.DATASETS} for model in plot.MODELS}
        path = root / 'old_curves.json'
        path.write_text(json.dumps(original))
        return path, pretrained, supplemental, manifests, original, current

    def test_all_six_curves_are_current_and_other_models_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            path, pretrained, supplemental, _, original, current = self.fixture(Path(temp))
            _, curves, sources = plot.collect_curves(path, pretrained, supplemental)
            self.assertEqual(set(sources), set(plot.DATASETS))
            for dataset in plot.DATASETS:
                self.assertEqual(curves['scGPT'][dataset], current)
            for model in plot.MODELS:
                if model != 'scGPT': self.assertEqual(curves[model], original[model])
            self.assertEqual(json.loads(path.read_text()), original)

    def test_missing_current_mdc_has_no_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            path, pretrained, _, _, _, _ = self.fixture(Path(temp))
            with self.assertRaisesRegex(ValueError, 'Missing current scGPT result'):
                plot.collect_curves(path, pretrained, None)

    def test_mismatched_mdc_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path, pretrained, supplemental, manifests, _, _ = self.fixture(Path(temp))
            manifest = manifests[supplemental]
            manifest['model_sha256'] = 'different-model'
            (supplemental / 'seed_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'loaded model differs'):
                plot.collect_curves(path, pretrained, supplemental)

    def test_curve_endpoint_must_reproduce_gene_level_score(self):
        with tempfile.TemporaryDirectory() as temp:
            path, pretrained, supplemental, manifests, _, _ = self.fixture(Path(temp))
            manifest = manifests[supplemental]
            manifest['metrics']['mDC']['accuracy_curve'] = [.5] * 16
            (supplemental / 'seed_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'Last step differs'):
                plot.collect_curves(path, pretrained, supplemental)


if __name__ == '__main__':
    unittest.main()
