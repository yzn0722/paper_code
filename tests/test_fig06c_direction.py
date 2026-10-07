"""Regression checks for Fig. 6c direction and same-run null provenance."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fig06c_plot', ROOT / 'plots/fig06c_plot_density_hESC.py')
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def reports():
    result = {}
    for index, (key, _, _) in enumerate(plot.SERIES):
        rho = np.arange(8, dtype=float) / 100 + index / 10
        null = np.linspace(-.02, .02, 200) + index / 100
        block = {'per_lag': [{'spearman': float(v)} for v in rho],
                 'windows': {'transient': {'observed': {'spearman': float(np.median(rho)), 'n_lags': 8},
                    'rewired': {'spearman': {'null_values': null.tolist(),
                        'null_mean': float(null.mean()), 'empirical_p_greater': 1 / 201}}}}}
        legacy = copy.deepcopy(block)
        legacy['per_lag'] = [{'spearman': .9} for _ in range(8)]
        legacy['windows']['transient']['observed']['spearman'] = .9
        legacy['windows']['transient']['rewired']['spearman']['null_values'] = [.9] * 200
        result[key] = {'preflight': {'n_null': 200, 'trajectory_shape': [33, 910]},
                       'input_provenance': {'trajectory_sha256': {'early': 'same'},
                                            'expression_sha256': 'same', 'vocab_sha256': 'same'},
                       'analyses': {str(d): {'query_to_key_primary': {'early': copy.deepcopy(block)},
                                            'key_to_query_primary': {'early': copy.deepcopy(legacy)}}
                                    for d in plot.DENSITIES}}
    return result


class DirectionTests(unittest.TestCase):
    def test_none_keeps_medians_and_lag_spread_without_error_extents(self):
        data = reports()
        no_errors = plot.load_observed_with_lag_errors(Path('.'), 'none', data)
        with_sd = plot.load_observed_with_lag_errors(Path('.'), 'sd', data)
        np.testing.assert_array_equal(no_errors.transient_spearman, with_sd.transient_spearman)
        np.testing.assert_array_equal(no_errors.lag_std, with_sd.lag_std)
        self.assertTrue((no_errors.yerr_lo == 0).all())
        self.assertTrue((no_errors.yerr_hi == 0).all())
        self.assertTrue((no_errors.error_kind == 'none').all())

    def write_reports(self, directory, data):
        for key, report in data.items():
            (directory / f'weighted_grn_propagation_{key}.json').write_text(json.dumps(report), encoding='utf-8')

    def test_selects_forward_observed_and_null_even_when_legacy_present(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.write_reports(directory, reports())
            observed = plot.load_observed_with_lag_errors(directory, 'sd')
            null = plot.load_pooled_null(directory)
            self.assertEqual(len(observed), 15)
            self.assertAlmostEqual(observed.iloc[0].transient_spearman, .035)
            self.assertAlmostEqual(null.iloc[0].rewired_null_mean, .01)
            self.assertEqual(null.iloc[0].rewired_null_n_pooled, 600)
            self.assertLess(null.iloc[0].rewired_null_q95, .04)
            self.assertTrue((observed.orientation == 'query_to_key_primary').all())

    def test_rejects_legacy_only_results(self):
        data = reports()
        for r in data.values():
            for block in r['analyses'].values():
                block.pop('query_to_key_primary')
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.write_reports(directory, data)
            with self.assertRaisesRegex(ValueError, 'regenerate query-to-key'):
                plot.load_reports(directory)

    def test_rejects_null_with_missing_draw(self):
        data = reports()
        data['attn']['analyses']['1000']['query_to_key_primary']['early']['windows']['transient']['rewired']['spearman']['null_values'].pop()
        with self.assertRaisesRegex(ValueError, '200 finite rewired draws'):
            plot.load_pooled_null(Path('.'), data)

    def test_rejects_mixed_trajectory_inputs(self):
        data = reports()
        data['cos_hid']['input_provenance']['trajectory_sha256']['early'] = 'other'
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.write_reports(directory, data)
            with self.assertRaisesRegex(ValueError, 'different trajectory_sha256'):
                plot.load_reports(directory)

    def test_rejects_stale_observed_summary(self):
        data = reports()
        data['attn']['analyses']['1000']['query_to_key_primary']['early']['windows']['transient']['observed']['spearman'] = .8
        with self.assertRaisesRegex(ValueError, '!= published'):
            plot.load_observed_with_lag_errors(Path('.'), 'sd', data)

    def test_rejects_mismatched_lag_window(self):
        data = reports()
        data['attn']['analyses']['1000']['query_to_key_primary']['early']['windows']['transient']['observed']['n_lags'] = 9
        with self.assertRaisesRegex(ValueError, 'expected 8 transient lags'):
            plot.load_observed_with_lag_errors(Path('.'), 'sd', data)


if __name__ == '__main__':
    unittest.main()
