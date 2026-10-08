"""Validate current-source checks and non-mutating trajectory recording."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'plots'))
from fig05_scgpt_panels import load_pretrained
from test_pseudotime_validation import load_runner


class OffsetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.))

    def forward(self, src, values, src_key_padding_mask):
        return {'mlm_output': values + 10}


class FigureProtocolTests(unittest.TestCase):
    def test_recorder_cannot_mutate_scores_or_state(self):
        evaluator = load_runner('run_scgpt_gene_results.py')
        evaluator.GEN_ITERS = 2
        ids = torch.tensor([[1, 2, 0]])
        values = torch.tensor([[0., 1., 9.], [0., 3., 9.]])
        pad = ids.eq(0).expand(2, -1)
        initial = np.array([2., 9.], dtype=np.float32)
        truth = np.array([1., -1.], dtype=np.float32)
        args = (OffsetModel(), ids, values, pad, np.array([False, True, True]), initial, truth, np.array([0, 1]))
        plain_curve, plain_mean = evaluator.iterative_direction_accuracy(*args)
        states = []
        def record(iteration, mean):
            states.append(mean.copy())
            mean[:] = -999  # A client recorder must not corrupt scoring.
        curve, mean = evaluator.iterative_direction_accuracy(*args, trajectory_callback=record)
        np.testing.assert_equal(mean, plain_mean)
        self.assertEqual(curve, plain_curve)
        np.testing.assert_allclose(states, [[3., 9.], [4., 9.]], atol=1e-5)

    def fixture(self, folder):
        df = pd.DataFrame({'gene':['A','B','C','D','OOV'], 'is_mapped':[1,1,1,1,0],
                           'in_eval':[1,0,0,0,0], 'delta_true':[4,3,-2,-1,0],
                           'delta_pred':[1,1,-1,1,0]})
        path=folder/'mHSC-L_gene_result.csv'
        df.to_csv(path,index=False)
        manifest={'condition':'pretrained', 'protocol':{
            'ema_alpha':.9,'eps_dir':.001,
            'input_processing':'scgpt.preprocess.binning per cell on mapped genes'},
            'metrics':{'mHSC-L':{'csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                 'balanced_accuracy_top30':1.}}}
        (folder/'seed_manifest.json').write_text(json.dumps(manifest))
        return manifest,path

    def test_changed_prediction_csv_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t); manifest,path=self.fixture(folder)
            load_pretrained(folder,'mHSC-L')
            path.write_text(path.read_text().replace('A,1,1,4,1','A,1,1,4,-1'))
            with self.assertRaisesRegex(ValueError,'checksum'):
                load_pretrained(folder,'mHSC-L')

    def test_old_ema_protocol_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t); manifest,_=self.fixture(folder)
            manifest['protocol']['ema_alpha']=.1
            (folder/'seed_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'EMA=0.9'):
                load_pretrained(folder,'mHSC-L')

    def test_oov_eval_gene_is_rejected_even_with_matching_hash(self):
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t); manifest,path=self.fixture(folder)
            df=pd.read_csv(path); df.in_eval=[0,0,0,0,1]; df.to_csv(path,index=False)
            manifest['metrics']['mHSC-L']['csv_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            (folder/'seed_manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'evaluation set'):
                load_pretrained(folder,'mHSC-L')


if __name__=='__main__':
    unittest.main()
