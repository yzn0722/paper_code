"""Regression tests for matched Fig. 4a curve and random candidate spaces."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'plots'))
spec = importlib.util.spec_from_file_location('fig04a', ROOT / 'plots/fig04a_edge_precision_hESC.py')
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


class CandidateTests(unittest.TestCase):
    def inputs(self, directory):
        gt = directory / 'gt.csv'
        pred = directory / 'pred.tsv'
        # A single TF, three reference genes; duplicates and self-only node must not count.
        pd.DataFrame([('A', 'B'), ('A', 'C'), ('A', 'B'), ('X', 'X')],
                     columns=['Gene1', 'Gene2']).to_csv(gt, index=False)
        pd.DataFrame([('A', 'B', .9), ('A', 'B', .8), ('B', 'A', .7),
                      ('A', 'C', .6), ('A', 'X', .5), ('A', 'A', .4)],
                     columns=['Gene1', 'Gene2', 'EdgeWeight']).to_csv(pred, sep='\t', index=False)
        return gt, pred

    def test_tf_curve_uses_same_unique_pairs_as_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            gt, pred = self.inputs(Path(temp))
            frame, n_true, candidates = plot.calculate_method_precision(str(pred), str(gt), percents=[100])
            self.assertEqual(candidates, {('A', 'B'), ('A', 'C')})
            self.assertEqual(n_true, 2)
            self.assertEqual(frame.iloc[0].PredEdgesTotal, 2)
            self.assertEqual(frame.iloc[0].Precision, 1.0)

    def test_unrestricted_mode_has_all_six_ordered_pairs(self):
        with tempfile.TemporaryDirectory() as temp:
            gt, pred = self.inputs(Path(temp))
            frame, n_true, candidates = plot.calculate_method_precision(str(pred), str(gt), TFEdges=False,
                                                                         percents=[100])
            self.assertEqual(len(candidates), 6)
            self.assertEqual(n_true, 2)
            self.assertEqual(frame.iloc[0].PredEdgesTotal, 3)
            self.assertAlmostEqual(frame.iloc[0].Precision, 2 / 3, places=6)

    def test_plot_line_and_export_use_tf_candidate_density(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            gt, pred = self.inputs(directory)
            csv_path = directory / 'curve.csv'
            with patch.object(plot.plt, 'axhline') as horizontal, patch.object(plot.plt, 'savefig'):
                plot.plot_three_methods_grn_curve([('emb500', str(pred))], str(gt), percents=[100],
                    save_data_path=str(csv_path), save_fig_path=str(directory / 'unused.pdf'))
            self.assertEqual(horizontal.call_args.kwargs['y'], 1.0)
            exported = pd.read_csv(csv_path)
            self.assertEqual(exported.iloc[0].CandidateEdgesTotal, 2)
            self.assertEqual(exported.iloc[0].RandomPrecision, 1.0)
            # The old all-gene denominator was 6 and would give an incorrect 1/3.


if __name__ == '__main__':
    unittest.main()
