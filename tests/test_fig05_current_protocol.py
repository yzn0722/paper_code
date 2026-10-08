"""Validate current-source checks and non-mutating trajectory recording."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
import io

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'plots'))
from fig05_scgpt_panels import load_pretrained, validate_full_group_early_curve
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

    def test_sampled_initial_state_run_is_rejected(self):
        report = {'iterations': 2, 'selected_cells': {'early': ['e1'], 'middle': ['m1'], 'late': ['l1']},
                  'group_available_sizes': {'early': 2, 'middle': 1, 'late': 1},
                  'balanced_accuracy_curves': {'early': [.5, .75]}}
        manifest = {'metrics': {'mHSC-L': {'accuracy_curve': [.5, .75]}}}
        with self.assertRaisesRegex(ValueError, 'every cell'):
            validate_full_group_early_curve(report, manifest, 'mHSC-L')

    def test_different_early_curve_is_rejected(self):
        report = {'iterations': 2, 'selected_cells': {'early': ['e1'], 'middle': ['m1'], 'late': ['l1']},
                  'group_available_sizes': {'early': 1, 'middle': 1, 'late': 1},
                  'balanced_accuracy_curves': {'early': [.5, .70]}}
        manifest = {'metrics': {'mHSC-L': {'accuracy_curve': [.5, .75]}}}
        with self.assertRaisesRegex(ValueError, 'match formal'):
            validate_full_group_early_curve(report, manifest, 'mHSC-L')

    def test_initial_state_runner_uses_full_groups_and_reproduces_formal_run(self):
        import run_scgpt_initial_state as runner
        evaluator = load_runner('run_scgpt_gene_results.py')
        evaluator.GEN_ITERS = 3
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            modeldir=root/'model'; modeldir.mkdir()
            (modeldir/'args.json').write_text('{"n_bins":51}')
            vocab={'<pad>':0, '<cls>':1, 'A':2, 'B':3}
            (modeldir/'vocab.json').write_text(json.dumps(vocab))
            (modeldir/'best_model.pt').write_bytes(b'explicit-fixture-checkpoint')
            cells=[f'cell{i}' for i in range(100)]
            expr=pd.DataFrame([np.linspace(1,40,100), np.linspace(40,1,100), np.full(100,999.)],
                              index=['A','B','OOV'],columns=cells)
            expr.to_csv(root/'expr.csv')
            pd.DataFrame({'cell':cells,'pt':np.arange(100)}).to_csv(root/'pt.csv',index=False)
            cfg={'expr_csv':str(root/'expr.csv'),'pt_csv':str(root/'pt.csv')}
            evaluator.MODEL_DIR=str(modeldir)
            evaluator.DATASETS={'mHSC-L':cfg}
            pretrained=root/'pretrained'; pretrained.mkdir()
            model=OffsetModel(); model.checkpoint_load_report={'fixture':True}
            with redirect_stdout(io.StringIO()):
                curve,_=evaluator.run_dataset('mHSC-L',cfg,model,vocab,torch.device('cpu'),pretrained)
            protocol={'input_processing':'scgpt.preprocess.binning per cell on mapped genes',
                      'ema_alpha':.9,'eps_dir':.001,'pt_quantile':.2,'top_percent':30,
                      'no_log1p':True,'gen_iters':3,'batch_size':16,
                      'preprocessing_seed_by_dataset':{'mHSC-L':11},
                      'input_sha256':{'mHSC-L':{k:runner.sha256(v) for k,v in cfg.items()}},
                      'checkpoint_sha256':runner.sha256(modeldir/'best_model.pt'),
                      'args_sha256':runner.sha256(modeldir/'args.json'),
                      'vocab_sha256':runner.sha256(modeldir/'vocab.json')}
            manifest={'condition':'pretrained','protocol':protocol,'metrics':{
                'mHSC-L':{'csv_sha256':runner.sha256(pretrained/'mHSC-L_gene_result.csv'), 'accuracy_curve':curve}}}
            (pretrained/'seed_manifest.json').write_text(json.dumps(manifest))
            out=root/'new_states'
            with patch.dict(sys.modules, {'run_scgpt_gene_results':evaluator}), \
                 patch.object(evaluator,'build_model',return_value=(model,vocab)), \
                 patch.object(torch.cuda,'is_available',return_value=False), \
                 patch.object(sys,'argv',['runner','--pretrained-dir',str(pretrained),'--outdir',str(out)]), \
                 redirect_stdout(io.StringIO()):
                runner.main()
            report=json.loads((out/'initial_state_manifest.json').read_text())
            self.assertEqual({k:len(v) for k,v in report['selected_cells'].items()},
                             {'early':20,'middle':60,'late':20})
            self.assertEqual(report['selected_cells']['early'], cells[:20])
            self.assertEqual(report['balanced_accuracy_curves']['early'], curve)
            self.assertEqual(report['iterations'], 3)
            validate_full_group_early_curve(report,manifest,'mHSC-L')
            with np.load(out/'reference.npz',allow_pickle=False) as ref:
                self.assertEqual(ref['genes'].tolist(), ['A','B','OOV'])


if __name__=='__main__':
    unittest.main()
