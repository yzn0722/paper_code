#!/usr/bin/env python3
"""Recompute all Fig. 6c query-to-key points/nulls, then plot the matching JSONs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SERIES = (('attn', 'output_att500'), ('cos_tok', 'output_emb500'),
          ('cos_hid', 'output_embhidden500'))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trajectory-dir', type=Path, required=True,
                   help='Directory containing the saved hESC early_mean_trajectory.npy.')
    p.add_argument('--benchmark-root', type=Path, default=Path('/mnt/10T/yzn/benchmark_GRN'))
    p.add_argument('--expression-csv', type=Path)
    p.add_argument('--vocab-json', type=Path)
    p.add_argument('--outdir', type=Path, default=ROOT / 'outputs/fig06c/query_to_key')
    p.add_argument('--seed', type=int, default=20260903)
    p.add_argument('--error', choices=('sd', 'sem', 'iqr'), default='sd')
    return p.parse_args()


def main():
    args = parse_args()
    expr = args.expression_csv or args.benchmark_root / 'input_process/CHIP/hESC_chip_matched-ExpressionData.csv'
    vocab = args.vocab_json or args.benchmark_root / 'pre_scgpt/scGPT/scgpt_human/vocab.json'
    raw = args.outdir / 'raw'
    networks = {}
    for key, folder in SERIES:
        candidates = [p for p in (args.benchmark_root / 'evl_omipath' / folder / 'scgpt').glob('*.tsv')
                      if p.stem.lower() == 'scgpt_hesc']
        if len(candidates) != 1:
            raise FileNotFoundError(f'{key}: expected one hESC network, found {candidates}')
        networks[key] = candidates[0]
    for path in (expr, vocab, args.trajectory_dir / 'early_mean_trajectory.npy'):
        if not path.is_file():
            raise FileNotFoundError(path)
    raw.mkdir(parents=True, exist_ok=True)
    commands = []
    for index, (key, _) in enumerate(SERIES):
        rep_out = raw / key
        command = [sys.executable, str(ROOT / 'upstream/dynamics/run_weighted_grn_propagation.py'),
                   '--trajectory-dir', str(args.trajectory_dir), '--expression-csv', str(expr),
                   '--vocab-json', str(vocab), '--grn-tsv', str(networks[key]),
                   '--network-name', key, '--outdir', str(rep_out),
                   '--top-k', '1000,5000,10000,20000,30000', '--groups', 'early',
                   '--n-null', '200', '--transient-iters', '8',
                   '--seed', str(args.seed + index * 1009), '--primary-only', '--execute']
        commands.append(command)
        print(f'Recomputing {key}: query-to-key, 5 densities, 200 nulls per density', flush=True)
        subprocess.run(command, check=True)
        shutil.copyfile(rep_out / 'weighted_grn_propagation.json',
                        raw / f'weighted_grn_propagation_{key}.json')
    (raw / 'run_commands.json').write_text(json.dumps(commands, indent=2) + '\n', encoding='utf-8')
    subprocess.run([sys.executable, str(ROOT / 'plots/fig06c_plot_density_hESC.py'),
                    '--json-dir', str(raw), '--outdir', str(args.outdir), '--error', args.error], check=True)


if __name__ == '__main__':
    main()
