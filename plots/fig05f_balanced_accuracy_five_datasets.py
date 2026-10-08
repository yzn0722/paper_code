#!/usr/bin/env python3
"""Figure 5f: five-dataset, six-model top-30% balanced-accuracy bar chart.

Input is the archived six-dataset summary CSV. mDC is deliberately excluded
because the manuscript's Figure 5f compares the other five datasets. This
script does not recompute model predictions or infer missing values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fig4_palette import apply_fig4_style, model_color


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "source_data/fig05f_balanced_accuracy_all_models_top30.csv"
DEFAULT_OUTPUT = ROOT / "outputs/fig05f_balanced_accuracy_five_datasets.pdf"

MODELS = ("scCello", "scPRINT", "scGPT", "Geneformer", "LangCell", "scFoundation")
DATASETS = ("hESC", "hHep", "mHSC-E", "mHSC-GM", "mHSC-L")
EXCLUDED_DATASET = "mDC"

# Shared physical heights and typography with Fig. 5c/d/e; wider for five groups.
PAGE_SIZE_PT = (638.4, 326.2515563964844)
AXES_BOUNDS_PT = (53.303125, 40.13125, 573.6, 277.2)


def read_balanced_accuracy(path: Path) -> dict[str, dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if not fields:
            raise ValueError("Empty CSV header")
        model_field = fields[0]
        expected_columns = set(DATASETS) | {EXCLUDED_DATASET}
        if set(fields[1:]) != expected_columns:
            raise ValueError(f"Expected dataset columns {sorted(expected_columns)}, found {fields[1:]}")
        table: dict[str, dict[str, float]] = {}
        for row in reader:
            model = (row[model_field] or "").strip()
            if model not in MODELS:
                raise ValueError(f"Unexpected model: {model!r}")
            if model in table:
                raise ValueError(f"Duplicate model: {model}")
            scores: dict[str, float] = {}
            for dataset in fields[1:]:
                value = float(row[dataset])
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(f"Invalid Balanced Accuracy for {model}/{dataset}: {value}")
                scores[dataset] = value
            table[model] = scores
    missing = set(MODELS) - set(table)
    if missing:
        raise ValueError(f"Missing models: {sorted(missing)}")
    return table


def write_used_source_data(path: Path, table: dict[str, dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", *DATASETS])
        for model in MODELS:
            writer.writerow([model, *(f"{table[model][ds]:.15g}" for ds in DATASETS)])


def plot(table: dict[str, dict[str, float]], output: Path) -> None:
    apply_fig4_style()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 14,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": None,
    })
    page_width, page_height = PAGE_SIZE_PT
    left, bottom, axes_width, axes_height = AXES_BOUNDS_PT
    fig = plt.figure(figsize=(page_width / 72, page_height / 72))
    ax = fig.add_axes([left / page_width, bottom / page_height,
                      axes_width / page_width, axes_height / page_height])
    x = list(range(len(DATASETS)))
    slot = 0.126
    width = 0.115
    for index, model in enumerate(MODELS):
        offset = (index - (len(MODELS) - 1) / 2) * slot
        ax.bar(
            [item + offset for item in x],
            [table[model][dataset] * 100 for dataset in DATASETS],
            width=width,
            color=model_color(model),
            edgecolor="none",
            label=model,
            zorder=2,
        )

    ax.axhline(50, color="#888888", linestyle="--", linewidth=0.9, zorder=1)
    ax.set_xlim(-0.56, len(DATASETS) - 0.44)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Balanced Accuracy (%)")
    ax.set_xticks(x, DATASETS)
    ax.tick_params(axis="both", length=0)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=len(MODELS),
        frameon=False,
        columnspacing=0.75,
        handlelength=1.0,
        handletextpad=0.25,
        borderpad=0,
        borderaxespad=0,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    scale = 72 / fig.dpi
    layout = {"page_size_pt": (fig.get_size_inches() * 72).tolist(),
              "axes_bounds_pt": [[v * scale for v in ax.get_window_extent().bounds]],
              "reference_panels": "Fig. 5c/d/e",
              "font_family": "DejaVu Sans", "tick_and_legend_font_pt": 14,
              "axis_label_font_pt": 16, "legend_position": "below, single row"}
    output.with_suffix(".layout.json").write_text(json.dumps(layout, indent=2) + "\n")
    fig.savefig(output, metadata={"Title": "Figure 5f: five-dataset balanced accuracy"})
    fig.savefig(output.with_suffix(".png"), dpi=600)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".pdf":
        parser.error("--output must end in .pdf")
    table = read_balanced_accuracy(args.input)
    plot(table, args.output)
    write_used_source_data(args.output.with_name(args.output.stem + "_source_data.csv"), table)
    print(f"input_sha256={hashlib.sha256(args.input.read_bytes()).hexdigest()}")
    print(f"models={','.join(MODELS)}")
    print(f"datasets={','.join(DATASETS)}; excluded={EXCLUDED_DATASET}")
    print(f"pdf={args.output}")


if __name__ == "__main__":
    main()
