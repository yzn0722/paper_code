#!/usr/bin/env python3
"""SI Fig. 3: update six scGPT refinement curves from current validated runs.

Other model curves are preserved from the supplied JSON. Display the original
11 steps by default; export every supplied step to the source-data CSV.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from audit_panel_alignment import require_matplotlib_panel_alignment
from fig05_scgpt_panels import load_pretrained, file_sha
from fig4_palette import apply_fig4_style, model_color

DATASETS = ('hESC', 'hHep', 'mDC', 'mHSC-E', 'mHSC-GM', 'mHSC-L')
MODELS = ('Geneformer', 'LangCell', 'scGPT', 'scFoundation', 'scPRINT', 'scCello')
PAGE_SIZE_PT = (344.80938720703125, 326.2515563964844)
AXES_BOUNDS_PT = (53.303125, 40.13125, 279.0, 277.2)


def collect_curves(original_path, pretrained, supplemental):
    original = json.loads(original_path.read_text())
    if set(original) != set(MODELS):
        raise ValueError('Expected all six model curves')
    curves = json.loads(json.dumps(original))
    reference = json.loads((pretrained / 'seed_manifest.json').read_text())
    sources = {}
    protocol_keys = ('version', 'input_processing', 'pt_quantile', 'top_percent',
                     'gen_iters', 'ema_alpha', 'no_log1p', 'eps_dir',
                     'checkpoint_sha256', 'args_sha256', 'vocab_sha256')
    for dataset in DATASETS:
        folder = pretrained if (pretrained / (dataset + '_gene_result.csv')).exists() else supplemental
        if folder is None:
            raise ValueError('Missing current scGPT result: ' + dataset)
        manifest, frame = load_pretrained(folder, dataset)
        protocol = manifest['protocol']
        if (protocol['gen_iters'] != 16 or protocol['pt_quantile'] != .2
                or protocol['top_percent'] != 30 or not protocol['no_log1p']):
            raise ValueError('Expected current native-binning 16-step scGPT protocol')
        if manifest['model_sha256'] != reference['model_sha256']:
            raise ValueError('scGPT loaded model differs across datasets')
        if any(protocol[key] != reference['protocol'][key] for key in protocol_keys):
            raise ValueError('Protocol mismatch: ' + dataset)
        metric = manifest['metrics'][dataset]
        curve = np.asarray(metric['accuracy_curve'], dtype=float)
        if curve.shape != (16,) or not np.isfinite(curve).all() or ((curve < 0) | (curve > 1)).any():
            raise ValueError('Invalid current scGPT curve: ' + dataset)
        if not np.isclose(curve[-1], metric['balanced_accuracy_top30'], rtol=0, atol=1e-12):
            raise ValueError('Last step differs from checksum-validated gene-level BA')
        curves['scGPT'][dataset] = curve.tolist()
        sources[dataset] = {
            'manifest_sha256': file_sha(folder / 'seed_manifest.json'),
            'gene_csv_sha256': file_sha(folder / (dataset + '_gene_result.csv')),
            'model_sha256': manifest['model_sha256'], 'protocol': protocol,
            'mapped_genes': int(frame.is_mapped.sum()),
            'top30_genes': int(frame.in_eval.sum()),
            'iteration16_balanced_accuracy': float(curve[-1]),
        }
    for model in MODELS:
        if set(curves[model]) != set(DATASETS):
            raise ValueError('Expected all six datasets: ' + model)
        for dataset in DATASETS:
            curve = np.asarray(curves[model][dataset], dtype=float)
            if curve.shape != (16,) or not np.isfinite(curve).all() or ((curve < 0) | (curve > 1)).any():
                raise ValueError('Invalid 16-step curve: ' + model + '/' + dataset)
        if model != 'scGPT' and curves[model] != original[model]:
            raise AssertionError('Non-scGPT curve changed')
    return original, curves, sources


def draw_panel(ax, curves, dataset, iterations, legend=False):
    for model in MODELS:
        y = np.asarray(curves[model][dataset][:iterations], dtype=float) * 100
        ax.plot(np.arange(1, iterations + 1), y, marker='o', lw=2.4,
                ms=8, alpha=.85, color=model_color(model), label=model)
    ax.set(xlim=(.8, iterations + .2), ylim=(0, 100),
           xlabel='Iteration', ylabel='Balanced accuracy (%)')
    ax.set_xticks(np.arange(1, iterations + 1, 2))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.axhline(50, color='gray', ls='--', lw=1.2, alpha=.8, zorder=0)
    ax.grid(False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=0)
    if legend:
        ax.legend(loc='lower right', frameon=False, ncol=2, fontsize=14,
                  handlelength=1.6, handletextpad=.5, labelspacing=.3,
                  borderaxespad=.3, columnspacing=.8)


def export(fig, folder, name):
    fig.canvas.draw()
    scale = 72 / fig.dpi
    layout = {'page_size_pt': (fig.get_size_inches() * 72).tolist(),
              'axes_bounds_pt': [[v * scale for v in ax.get_window_extent().bounds]
                                 for ax in fig.axes]}
    (folder / (name + '.layout.json')).write_text(json.dumps(layout, indent=2) + '\n')
    fig.savefig(folder / (name + '.pdf'), dpi=600)
    fig.savefig(folder / (name + '.svg'), dpi=600)
    fig.savefig(folder / (name + '.png'), dpi=600)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curves-json', type=Path, required=True)
    parser.add_argument('--pretrained-dir', type=Path, required=True)
    parser.add_argument('--supplemental-dir', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=11)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 16:
        parser.error('--iterations must be between 1 and 16')
    original, curves, sources = collect_curves(args.curves_json, args.pretrained_dir, args.supplemental_dir)
    args.outdir.mkdir(parents=True, exist_ok=True)
    single = args.outdir / 'individual'
    single.mkdir(exist_ok=True)
    apply_fig4_style()
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                        'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 16,
                        'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14,
                        'pdf.fonttype': 42, 'svg.fonttype': 'none',
                        'savefig.bbox': None, 'figure.dpi': 100})
    for dataset in DATASETS:
        width, height = PAGE_SIZE_PT
        left, bottom, aw, ah = AXES_BOUNDS_PT
        fig = plt.figure(figsize=(width / 72, height / 72))
        ax = fig.add_axes([left / width, bottom / height, aw / width, ah / height])
        draw_panel(ax, curves, dataset, args.iterations, legend=True)
        export(fig, single, 'convergence_models_' + dataset + '_balanced_accuracy')
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.subplots_adjust(left=.09, right=.98, bottom=.16, top=.94, wspace=.30, hspace=.35)
    for dataset, ax in zip(DATASETS, axes.flat):
        draw_panel(ax, curves, dataset, args.iterations)
        ax.set_title(dataset, pad=12)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .025),
               ncol=6, frameon=False, fontsize=14, handlelength=1.6,
               handletextpad=.5, columnspacing=.8)
    require_matplotlib_panel_alignment(fig, json_out=args.outdir / 'panel_alignment.json')
    export(fig, args.outdir, 'Supplementary_Fig_3_balanced_convergence_six_datasets')
    (args.outdir / 'balanced_accuracy_curves_all_models.json').write_text(json.dumps(curves, indent=2) + '\n')
    with (args.outdir / 'Supplementary_Fig_3_source_data.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['dataset', 'model', 'iteration', 'balanced_accuracy', 'displayed', 'source'])
        for dataset in DATASETS:
            for model in MODELS:
                for i, value in enumerate(curves[model][dataset], 1):
                    writer.writerow([dataset, model, i, repr(value), int(i <= args.iterations),
                                     'current scGPT manifest' if model == 'scGPT' else 'preserved previous curve'])
    report = {'complete': True, 'displayed_iterations': args.iterations,
              'available_iterations': 16, 'datasets': list(DATASETS),
              'original_curves_sha256': file_sha(args.curves_json),
              'current_scgpt_sources': sources,
              'non_scgpt_curves_unchanged': all(curves[m] == original[m] for m in MODELS if m != 'scGPT'),
              'all_scgpt_endpoints_validated_from_gene_csv': True,
              'non_scgpt_validation_scope': 'Preserved supplied curves; their model runs were not revalidated.'}
    (args.outdir / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: six current scGPT curves; other curves preserved; 16-step source data exported')


if __name__ == '__main__':
    main()
