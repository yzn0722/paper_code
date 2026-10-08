#!/usr/bin/env python3
"""Recompute three Fig. 5d starts with the validated formal scGPT protocol.

Requires the manifest and pretrained CSV from the fresh Fig. 6a experiment.
Uses all cells in each start group. The early trajectory must reproduce the
formal pretrained run exactly; the binned reference and top genes are shared.
This writes results to a new directory and does not update figures.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained-dir', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--dataset', default='mHSC-L')
    parser.add_argument('--iterations', type=int,
                        help='Defaults to the full iteration count in the pretrained manifest')
    args = parser.parse_args()
    manifest = json.loads((args.pretrained_dir / 'seed_manifest.json').read_text())
    protocol = manifest['protocol']
    if args.iterations is None:
        args.iterations = int(protocol['gen_iters'])
    if not 1 <= args.iterations <= int(protocol['gen_iters']):
        parser.error('iterations must be within the recorded pretrained run')
    if manifest.get('condition') != 'pretrained' or (
        protocol['input_processing'], protocol['ema_alpha'], protocol['eps_dir'],
        protocol['pt_quantile'], protocol['top_percent'], protocol['no_log1p']
    ) != ('scgpt.preprocess.binning per cell on mapped genes', .9, .001, .2, 30, True):
        raise ValueError('Requires a validated native-binning EMA=0.9 pretrained run')
    import run_scgpt_gene_results as evaluator
    ds = args.dataset
    reference_path = args.pretrained_dir / f'{ds}_gene_result.csv'
    if sha256(reference_path) != manifest['metrics'][ds]['csv_sha256']:
        raise ValueError('Pretrained result CSV checksum mismatch')
    for key in ('expr_csv', 'pt_csv'):
        if sha256(evaluator.DATASETS[ds][key]) != protocol['input_sha256'][ds][key]:
            raise ValueError(f'Input changed: {key}')
    for name, key in [('best_model.pt', 'checkpoint_sha256'), ('args.json', 'args_sha256'), ('vocab.json', 'vocab_sha256')]:
        if sha256(Path(evaluator.MODEL_DIR) / name) != protocol[key]:
            raise ValueError(f'Model changed: {name}')
    if (evaluator.EMA_ALPHA, evaluator.EPS_DIR, evaluator.TOP_PERCENT,
        evaluator.PT_QUANTILE, evaluator.BATCH_SIZE, evaluator.NO_LOG1P) != (
        protocol['ema_alpha'], protocol['eps_dir'], protocol['top_percent'],
        protocol['pt_quantile'], protocol['batch_size'], protocol['no_log1p']):
        raise ValueError('Formal evaluator parameters disagree with reference')
    seed = protocol['preprocessing_seed_by_dataset'][ds]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, vocab = evaluator.build_model(evaluator.MODEL_DIR, device)
    expr = pd.read_csv(evaluator.DATASETS[ds]['expr_csv'], index_col=0)
    expr, pt, pt_stats = evaluator.load_aligned_pseudotime(expr, evaluator.DATASETS[ds]['pt_csv'])
    early, late, _, _ = evaluator.split_pseudotime_quantiles(pt, evaluator.PT_QUANTILE)
    groups = {'early': early, 'middle': ~(early | late), 'late': late}
    genes = np.asarray(expr.index.astype(str), dtype=str)
    mapped = np.array([evaluator.convert_mouse_to_human_gene(g) in vocab for g in genes])
    bins = int(json.loads((Path(evaluator.MODEL_DIR) / 'args.json').read_text()).get('n_bins', 51))
    x = np.zeros((expr.shape[1], expr.shape[0]), dtype=np.float32)
    x[:, mapped] = evaluator.bin_expr_to_0_50(expr.T.to_numpy(np.float32)[:, mapped], do_log1p=False, n_bins=bins)
    early_mean, late_mean = x[early].mean(0), x[late].mean(0)
    reference = pd.read_csv(reference_path)
    if not np.array_equal(reference.gene.astype(str), genes) or not np.array_equal(reference.is_mapped.to_numpy(), mapped.astype(int)):
        raise ValueError('Reference gene order or mapping mismatch')
    if not np.allclose(reference.true_early_mean, early_mean, atol=1e-6, rtol=0) or not np.allclose(reference.true_late_mean, late_mean, atol=1e-6, rtol=0):
        raise ValueError('Binned reference differs from Fig. 6a; check preprocessing seed')
    truth = late_mean - early_mean
    pool = np.flatnonzero(mapped)
    top = pool[np.argsort(np.abs(truth[pool]))[::-1][:max(int(len(pool)*.30), 1)]]
    if set(top) != set(np.flatnonzero(reference.in_eval.to_numpy() == 1)):
        raise ValueError('Top-gene set differs from Fig. 6a')
    args.outdir.mkdir(parents=True, exist_ok=False)
    ids = torch.tensor([[vocab['<cls>']] + [vocab[evaluator.convert_mouse_to_human_gene(g)] if mapped[i] else vocab['<pad>'] for i, g in enumerate(genes)]], dtype=torch.long)
    values = torch.tensor(np.concatenate([np.zeros((len(pt), 1)), x], axis=1), dtype=torch.float16 if device.type == 'cuda' else torch.float32)
    pad = ids.eq(vocab['<pad>']).expand(len(pt), -1)
    update = np.r_[False, np.ones(len(genes), dtype=bool)]
    evaluator.GEN_ITERS = args.iterations
    selected, curves = {}, {}
    for name, mask in groups.items():
        idx = np.flatnonzero(mask)
        if not len(idx):
            raise ValueError(f'Empty starting group: {name}')
        selected[name] = expr.columns[idx].astype(str).tolist()
        initial = x[idx].mean(0)
        states = [initial.copy()]
        curve, final = evaluator.iterative_direction_accuracy(
            model, ids, values[idx], pad[idx], update, initial, truth, top,
            trajectory_callback=lambda iteration, mean: states.append(mean.astype(np.float32)))
        if name == 'early':
            expected = manifest['metrics'][ds]['accuracy_curve'][:args.iterations]
            if not np.array_equal(np.asarray(curve), np.asarray(expected)):
                raise RuntimeError('Early BA trajectory differs from formal Fig. 5c/scGPT reference')
            if args.iterations == int(protocol['gen_iters']) and not np.allclose(
                final, reference.pred_late_like_mean.to_numpy(float), atol=1e-6, rtol=0
            ):
                raise RuntimeError('Final early predictions differ from formal pretrained CSV')
        np.save(args.outdir / f'{name}_mean_trajectory.npy', np.asarray(states, dtype=np.float32))
        curves[name] = curve
        print(name, 'n_cells', len(idx), 'BA', curve, flush=True)
    np.savez_compressed(args.outdir / 'reference.npz', genes=genes, is_mapped=mapped,
                        true_delta=truth, top_idx=top, early_mean=early_mean, late_mean=late_mean)
    report = {'dataset': ds, 'protocol': protocol, 'iterations': args.iterations,
              'cell_selection': 'all cells in each pseudotime group; no subsampling',
              'early_matches_formal_curve': True,
              'runner_sha256': sha256(__file__),
              'evaluator_sha256': sha256(evaluator.__file__),
              'selected_cells': selected, 'group_available_sizes': {k: int(v.sum()) for k, v in groups.items()},
              'balanced_accuracy_curves': curves, 'reference_csv_sha256': sha256(reference_path),
              'pseudotime_filter': pt_stats, 'checkpoint_load_report': model.checkpoint_load_report,
              'definition': 'change from each starting group initial mean; fixed full early-to-late binned reference; per-cell updates then mean'}
    (args.outdir / 'initial_state_manifest.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
