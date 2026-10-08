#!/usr/bin/env python3
"""Replot Fig. 5c/d/e from explicit, checksum-validated current scGPT results.

Other model curves are preserved from --curves-json. Their preservation is not
validation of those model runs. Initial-state trajectories require the output
of run_scgpt_initial_state.py; legacy globally-scaled trajectories are rejected.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'upstream/dynamics'))
from direction_metrics import EPS_DIR, direction_scores
from fig4_palette import apply_fig4_style, model_color

ORDER = ['Geneformer', 'LangCell', 'scGPT', 'scFoundation', 'scPRINT', 'scCello']
# Fixed exported geometry and typography shared by the manuscript panels.
PAGE_SIZE_PT = (344.80938720703125, 326.2515563964844)
AXES_BOUNDS_PT = (53.303125, 40.13125, 279.0, 277.2)
STARTS = {'early': ('Early start', '#2E6F9E'),
          'middle': ('Intermediate start', '#D98B2B'),
          'late': ('Late start', '#3E9B76')}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_pretrained(folder, dataset):
    manifest = json.loads((folder / 'seed_manifest.json').read_text())
    p = manifest['protocol']
    if manifest.get('condition') != 'pretrained' or p.get('ema_alpha') != .9 or p.get('eps_dir') != EPS_DIR or p.get('input_processing') != 'scgpt.preprocess.binning per cell on mapped genes':
        raise ValueError('Use the validated native-binning EMA=0.9 pretrained experiment')
    csv = folder / f'{dataset}_gene_result.csv'
    if file_sha(csv) != manifest['metrics'][dataset]['csv_sha256']:
        raise ValueError('Pretrained CSV checksum mismatch')
    frame = pd.read_csv(csv)
    if not frame.is_mapped.isin([0, 1]).all() or not frame.in_eval.isin([0, 1]).all():
        raise ValueError('Mapping and evaluation flags must be binary')
    mapped = frame.is_mapped.to_numpy() == 1
    top = frame.in_eval.to_numpy() == 1
    if (top & ~mapped).any() or top.sum() != max(int(mapped.sum() * .3), 1):
        raise ValueError('Invalid mapped top-30% evaluation set')
    values = frame[['delta_true', 'delta_pred']].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite prediction or reference')
    ba = direction_scores(values[:, 1], values[:, 0], np.flatnonzero(top), EPS_DIR)['balanced_accuracy']
    if not np.isclose(ba, manifest['metrics'][dataset]['balanced_accuracy_top30'], atol=1e-12, rtol=0):
        raise ValueError('CSV does not reproduce manifest balanced accuracy')
    return manifest, frame


def new_axes(ylabel):
    # Preserve the legacy exported page and plot area, without variable cropping.
    width, height = PAGE_SIZE_PT
    left, bottom, axes_width, axes_height = AXES_BOUNDS_PT
    fig = plt.figure(figsize=(width / 72, height / 72))
    ax = fig.add_axes([left / width, bottom / height,
                      axes_width / width, axes_height / height])
    ax.set_xlabel('Iteration')
    ax.set_ylabel(ylabel)
    ax.spines[['top', 'right']].set_visible(False)
    return fig, ax


def export(fig, outdir, name):
    # Editable vector text; PNG is a high-resolution inspection preview.
    fig.canvas.draw()
    scale = 72 / fig.dpi
    layout = {'page_size_pt': (fig.get_size_inches() * 72).tolist(),
              'axes_bounds_pt': [[v * scale for v in ax.get_window_extent().bounds]
                                 for ax in fig.axes]}
    (outdir / f'{name}.layout.json').write_text(json.dumps(layout, indent=2) + '\n')
    fig.savefig(outdir / f'{name}.pdf', dpi=600)
    fig.savefig(outdir / f'{name}.svg', dpi=600)
    fig.savefig(outdir / f'{name}.png', dpi=600)
    plt.close(fig)


def panel_c(args, manifest):
    before = json.loads(args.curves_json.read_text())
    curves = json.loads(json.dumps(before))
    curves['scGPT'][args.dataset] = manifest['metrics'][args.dataset]['accuracy_curve']
    for name in before:
        if name != 'scGPT' and curves[name] != before[name]:
            raise AssertionError('Non-scGPT curve changed')
    rows = []
    fig, ax = new_axes('Balanced accuracy (%)')
    for name in ORDER:
        y = np.asarray(curves[name][args.dataset], float)[:args.c_iterations]
        if len(y) != args.c_iterations or not np.isfinite(y).all() or ((y < 0) | (y > 1)).any():
            raise ValueError(f'Invalid convergence curve: {name}')
        ax.plot(np.arange(1, len(y)+1), y*100, marker='o', lw=2.4, ms=8,
                alpha=.85, color=model_color(name), label=name)
        rows += [{'model': name, 'iteration': i+1, 'balanced_accuracy': float(value),
                  'source': 'current pretrained manifest' if name == 'scGPT' else 'preserved previous curve'} for i, value in enumerate(y)]
    ax.set(xlim=(.8, args.c_iterations+.2), ylim=(0, 100))
    ax.set_xticks(np.arange(1, args.c_iterations+1, 2))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.axhline(50, color='gray', ls='--', lw=1.2, alpha=.8, zorder=0)
    ax.legend(loc='lower right', frameon=False, ncol=2, fontsize=14,
              handlelength=1.6, handletextpad=.5, labelspacing=.3,
              borderaxespad=.3, columnspacing=.8)
    export(fig, args.outdir, 'fig05c_convergence_mHSC-L')
    pd.DataFrame(rows).to_csv(args.outdir/'fig05c_source_data.csv', index=False)
    (args.outdir/'updated_curves_all_models.json').write_text(json.dumps(curves, indent=2)+'\n')
    return {'shown_iterations': args.c_iterations, 'new_scgpt_curve': curves['scGPT'][args.dataset],
            'old_scgpt_curve': before['scGPT'][args.dataset],
            'other_model_curves_unchanged': True}


def panel_d(args, manifest, frame):
    report = json.loads((args.trajectory_dir/'initial_state_manifest.json').read_text())
    if report['dataset'] != args.dataset or report['protocol'] != manifest['protocol'] or report['reference_csv_sha256'] != manifest['metrics'][args.dataset]['csv_sha256']:
        raise ValueError('Initial-state run and Fig. 6a reference have different protocols')
    validate_full_group_early_curve(report, manifest, args.dataset)
    shown_iterations = min(args.c_iterations, report['iterations'])
    if shown_iterations < 1:
        raise ValueError('At least one iteration must be displayed')
    with np.load(args.trajectory_dir/'reference.npz', allow_pickle=False) as ref:
        truth, top, genes = ref['true_delta'], ref['top_idx'], ref['genes']
    if not np.array_equal(genes.astype(str), frame.gene.astype(str)) or not np.allclose(truth, frame.delta_true, atol=1e-6, rtol=0) or set(top) != set(np.flatnonzero(frame.in_eval.to_numpy()==1)):
        raise ValueError('Initial-state reference genes or truth differ from Fig. 6a')
    rows, curves = [], {}
    fig, ax = new_axes('Balanced accuracy (%)')
    for key, (label, color) in STARTS.items():
        states = np.load(args.trajectory_dir/f'{key}_mean_trajectory.npy')
        if states.shape != (report['iterations']+1, len(truth)) or not np.isfinite(states).all():
            raise ValueError(f'Invalid trajectory: {key}')
        y = []
        for i in range(1, shown_iterations + 1):
            ba = direction_scores(states[i]-states[0], truth, top, EPS_DIR)['balanced_accuracy']
            step_ba = direction_scores(states[i]-states[i-1], truth, top, EPS_DIR)['balanced_accuracy']
            y.append(ba)
            rows.append({'start': label, 'iteration': i, 'balanced_accuracy': ba,
                         'stepwise_balanced_accuracy': step_ba, 'n_cells': len(report['selected_cells'][key]),
                         'n_top_genes': len(top)})
        if not np.allclose(y, report['balanced_accuracy_curves'][key][:shown_iterations], atol=1e-12, rtol=0):
            raise ValueError(f'Saved trajectory cannot reproduce evaluator BA: {key}')
        curves[key] = y
        ax.plot(np.arange(1, len(y)+1), np.asarray(y)*100, marker='o',
                lw=2.4, ms=8, markeredgewidth=0, alpha=.85, color=color, label=label)
    ax.set(xlim=(.8, shown_iterations+.2), ylim=(0, 100))
    ax.set_xticks(np.arange(1, shown_iterations+1, 2))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.axhline(50, color='gray', ls='--', lw=1.2, alpha=.8, zorder=0)
    ax.legend(loc='lower right', frameon=False, fontsize=14,
              handlelength=1.6, handletextpad=.5, labelspacing=.3,
              borderaxespad=.3, columnspacing=.8)
    export(fig, args.outdir, 'fig05d_initial_state_mHSC-L')
    pd.DataFrame(rows).to_csv(args.outdir/'fig05d_source_data.csv', index=False)
    return {'curves': curves, 'shown_iterations': shown_iterations,
            'early_matches_fig05c_scgpt': True,
            'n_top_genes': len(top), 'selected_cells': report['selected_cells'],
            'definition': report['definition']}


def validate_full_group_early_curve(report, manifest, dataset):
    """Reject sampled or inconsistent d inputs rather than copying a c curve."""
    for key in STARTS:
        cells = report['selected_cells'][key]
        if len(set(cells)) != len(cells) or len(cells) != report['group_available_sizes'][key]:
            raise ValueError(f'Panel d requires every cell in the {key} group; sampled runs are incompatible with c')
    n = report['iterations']
    actual = np.asarray(report['balanced_accuracy_curves']['early'], dtype=float)
    expected = np.asarray(manifest['metrics'][dataset]['accuracy_curve'], dtype=float)[:n]
    if len(actual) != n or not np.array_equal(actual, expected):
        raise ValueError('Panel d Early curve must match formal Fig. 5c/scGPT exactly')


def panel_e(args, manifest, frame):
    # The scatter describes continuous changes in all mapped genes, whereas the
    # BA panels use the top 30%. Near-zero true directions remain visible in grey.
    mapped = frame.is_mapped.to_numpy() == 1
    x, y = frame.delta_true.to_numpy(float)[mapped], frame.delta_pred.to_numpy(float)[mapped]
    true_dir = np.where(x > EPS_DIR, 1, np.where(x < -EPS_DIR, -1, 0))
    pred_dir = np.where(y > EPS_DIR, 1, np.where(y < -EPS_DIR, -1, 0))
    consistent = (true_dir != 0) & (true_dir == pred_dir)
    inconsistent = (true_dir != 0) & ~consistent
    neutral = true_dir == 0
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        raise ValueError('Pearson correlation requires variable mapped genes')
    r, p = pearsonr(x, y)
    fig, ax = new_axes('Predicted expression change')
    ax.set_xlabel('Observed expression change')
    for mask, label, color in [(consistent, 'Consistent', model_color('scGPT')),
                                (inconsistent, 'Inconsistent', model_color('scPrint')),
                                (neutral, 'Near-zero observed', '#9C9C9C')]:
        if mask.any():
            ax.scatter(x[mask], y[mask], s=60, color=color, alpha=.75,
                       edgecolors='none', linewidths=0, label=label, zorder=2)
    fit = np.polyfit(x, y, 1)
    grid = np.linspace(x.min(), x.max(), 100)
    ax.plot(grid, fit[0]*grid+fit[1], color='black', lw=1.2, alpha=.7, zorder=1)
    ax.axhline(0, color='gray', ls=':', lw=.8, alpha=.7, zorder=0)
    ax.axvline(0, color='gray', ls=':', lw=.8, alpha=.7, zorder=0)
    ax.margins(x=.08, y=.12)
    ax.text(.95, .95, f'r = {r:.3f}', transform=ax.transAxes, ha='right', va='top', fontsize=14)
    ax.legend(loc='lower right', frameon=False, fontsize=14,
              handletextpad=.4, borderpad=.2, labelspacing=.3)
    export(fig, args.outdir, 'fig05e_gene_change_mHSC-L')
    source = frame.loc[mapped].copy()
    source['display_direction_class'] = np.where(neutral, 'Near-zero observed', np.where(consistent, 'Consistent', 'Inconsistent'))
    source.to_csv(args.outdir/'fig05e_source_data.csv', index=False)
    return {'pearson_r': float(r), 'pearson_p_unadjusted': float(p), 'n_mapped_genes': int(mapped.sum()),
            'n_oov_excluded': int((~mapped).sum()), 'n_near_zero_true': int(neutral.sum()),
            'n_consistent': int(consistent.sum()), 'n_inconsistent': int(inconsistent.sum()),
            'prediction_iterations': manifest['protocol']['gen_iters'],
            'pearson_scope': 'all mapped genes; genes are not independent biological replicates; P is descriptive'}


def main(panel=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', choices=['c', 'd', 'e', 'all'], default=panel or 'all')
    parser.add_argument('--pretrained-dir', type=Path, required=True)
    parser.add_argument('--curves-json', type=Path)
    parser.add_argument('--trajectory-dir', type=Path)
    parser.add_argument('--dataset', default='mHSC-L')
    parser.add_argument('--c-iterations', type=int, default=11)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    if args.dataset != 'mHSC-L':
        parser.error('These main Fig. 5 panel entry points are for mHSC-L')
    if args.panel in ('c', 'all') and args.curves_json is None:
        parser.error('--curves-json is required for panel c')
    if args.panel in ('d', 'all') and args.trajectory_dir is None:
        parser.error('--trajectory-dir is required for panel d')
    args.outdir.mkdir(parents=True, exist_ok=True)
    apply_fig4_style()
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'svg.fonttype': 'none', 'savefig.bbox': None,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'savefig.facecolor': 'white', 'savefig.edgecolor': 'white'})
    manifest, frame = load_pretrained(args.pretrained_dir, args.dataset)
    stats = {'dataset': args.dataset, 'protocol': manifest['protocol'],
             'pretrained_model_sha256': manifest['model_sha256'],
             'display_style': {'page_size_pt': PAGE_SIZE_PT, 'axes_bounds_pt': AXES_BOUNDS_PT,
                               'font_family': plt.rcParams['font.family'],
                               'axis_label_pt': 16, 'tick_label_pt': 14, 'legend_pt': 14,
                               'axis_linewidth_pt': 1.2, 'preview_dpi': 600}}
    if args.panel in ('c', 'all'):
        stats['c'] = panel_c(args, manifest)
    if args.panel in ('d', 'all'):
        stats['d'] = panel_d(args, manifest, frame)
    if args.panel in ('e', 'all'):
        stats['e'] = panel_e(args, manifest, frame)
    (args.outdir/f'fig05{args.panel}_provenance.json').write_text(json.dumps(stats, indent=2)+'\n')
    print(json.dumps({k: v for k, v in stats.items() if k in ('c', 'd', 'e')}, indent=2))


if __name__ == '__main__':
    main()
