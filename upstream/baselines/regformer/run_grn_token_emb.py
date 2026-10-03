#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RegFormer GRN from pretrained token embeddings (no mamba_ssm / GPU required).

Uses encoder.embedding.weight from best_model.pt + cosine similarity,
TF→target scoring, exports Gene1/Gene2/EdgeWeight for FBEval.

NOTE: Official RegFormer GRN uses contextual gene embeddings from Mamba
forward (hidemb). This script is the emb-equivalent fallback when CUDA/mamba
is unavailable. Re-run with downstream_task/regformer_grn.py when GPU works.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import cosine_similarity

GT_SOURCES = {
    "STRING": ("STRING", "{dataset}_processed-network.csv"),
    "Non_CHIP": ("Non_CHIP", "{dataset}_processed-network.csv"),
    "CHIP": ("CHIP", "{dataset}_chip_matched-network.csv"),
    "omnipath": ("omnipath", "{dataset}_processed-network.csv"),
}


def norm(g: str) -> str:
    return str(g).strip().upper()


def load_vocab(path: Path) -> Dict[str, int]:
    raw = json.loads(path.read_text())
    # map both original and upper for matching
    out = {}
    for g, i in raw.items():
        out[str(g)] = int(i)
        out[norm(g)] = int(i)
    return out


def load_tfs(path: Path) -> Set[str]:
    df = pd.read_csv(path, header=None, sep="\t")
    return {norm(x) for x in df[0].tolist() if str(x).strip()}


def load_expr_genes(h5ad_path: Path) -> List[str]:
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path)
    return [str(g) for g in adata.var_names]


def load_gt(path: Path) -> Tuple[Set[str], Set[str], Set[Tuple[str, str]]]:
    df = pd.read_csv(path)
    df["Gene1"] = df["Gene1"].map(norm)
    df["Gene2"] = df["Gene2"].map(norm)
    df = df[(df.Gene1 != "") & (df.Gene2 != "") & (df.Gene1 != df.Gene2)]
    tfs = set(df.Gene1)
    genes = set(df.Gene1) | set(df.Gene2)
    edges = set(zip(df.Gene1, df.Gene2))
    return tfs, genes, edges


def evaluate(pred: pd.DataFrame, tfs: Set[str], genes: Set[str], true_edges: Set[Tuple[str, str]]):
    score = {
        (norm(a), norm(b)): float(w)
        for a, b, w in pred[["Gene1", "Gene2", "EdgeWeight"]].itertuples(index=False)
    }
    labels, scores = [], []
    for tf in tfs:
        for g in genes:
            if tf == g:
                continue
            labels.append(1 if (tf, g) in true_edges else 0)
            scores.append(score.get((tf, g), -1.0))
    y = np.asarray(labels)
    p = np.asarray(scores)
    n_possible = len(tfs) * len(genes) - len(tfs)
    n_true = len(true_edges)
    baseline = n_true / n_possible if n_possible else 0.0
    aupr = float(average_precision_score(y, p)) if y.sum() else 0.0
    ratio = aupr / baseline if baseline else 0.0
    order = np.argsort(-p)
    k = min(n_true, len(order))
    epr = ((y[order[:k]].sum() / k) / baseline) if k and baseline else 0.0
    return {
        "AUPR": round(aupr, 6),
        "AUPR_Ratio": round(ratio, 6),
        "EPR": round(float(epr), 6),
        "Random_Baseline": round(baseline, 8),
        "True_Edges": n_true,
        "Possible_Edges": n_possible,
    }


def build_edges(
    genes: Sequence[str],
    emb: np.ndarray,
    tf_set: Set[str],
    *,
    top_k: int = 0,
    min_sim: float = -1.0,
) -> pd.DataFrame:
    """
    If top_k>0: paper-style TF→top-k targets.
    If top_k==0: export all TF→gene cosine scores (better for AUPR).
    """
    names = [norm(g) for g in genes]
    sim = cosine_similarity(emb)
    rows = []
    name_to_i = {g: i for i, g in enumerate(names)}
    for g in names:
        if g not in tf_set:
            continue
        i = name_to_i[g]
        scores = sim[i].copy()
        scores[i] = -np.inf
        if top_k > 0:
            valid = np.where(scores > min_sim)[0]
            if len(valid) > top_k:
                valid = valid[np.argsort(scores[valid])[-top_k:]]
            idxs = valid
        else:
            idxs = np.arange(len(names))
            idxs = idxs[idxs != i]
        for j in idxs:
            rows.append((names[i], names[j], float(scores[j])))
    out = pd.DataFrame(rows, columns=["Gene1", "Gene2", "EdgeWeight"])
    out = out.sort_values("EdgeWeight", ascending=False).reset_index(drop=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=Path,
                   default=Path("/mnt/10T/yzn/RegFormer/checkpoints/extracted/RegFormer-10k"))
    p.add_argument("--data-dir", type=Path, default=Path("/mnt/10T/yzn/RegFormer/data"))
    p.add_argument("--gt-root", type=Path, default=Path("/mnt/10T/yzn/benchmark_GRN/input_process"))
    p.add_argument("--tf-file", type=Path, default=Path("/mnt/10T/yzn/RegFormer/resources/hs_hgnc_tfs.txt"))
    p.add_argument("--datasets", nargs="+",
                   default=["hESC", "hHep", "mDC", "mHSC-E", "mHSC-GM", "mHSC-L"])
    p.add_argument(
        "--gt-types",
        nargs="+",
        default=["STRING", "Non_CHIP", "CHIP", "omnipath"],
        choices=list(GT_SOURCES.keys()),
    )
    p.add_argument("--top-k", type=int, default=0, help="0=all TF-gene scores; >0=paper top-k")
    p.add_argument("--min-sim", type=float, default=-1.0)
    p.add_argument("--outdir", type=Path, default=Path("/mnt/10T/yzn/RegFormer/outputs/grn_token_emb"))
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing prediction TSVs and only re-evaluate.",
    )
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    pred_dir = args.outdir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    vocab = load_vocab(args.ckpt_dir / "vocab.json")
    id_to_gene = {}
    raw = json.loads((args.ckpt_dir / "vocab.json").read_text())
    for g, i in raw.items():
        id_to_gene[int(i)] = str(g)

    state = torch.load(args.ckpt_dir / "best_model.pt", map_location="cpu")
    W = state["encoder.embedding.weight"].detach().cpu().numpy()  # [V, D]
    print(f"token emb matrix: {W.shape}")

    tf_set = load_tfs(args.tf_file)
    print(f"TF list size: {len(tf_set)}")

    rows = []
    for ds in args.datasets:
        h5ad = args.data_dir / f"{ds}.h5ad"
        if not h5ad.is_file():
            print(f"[skip] missing {h5ad}")
            continue
        genes_raw = load_expr_genes(h5ad)
        # keep genes in vocab
        keep_genes, keep_idx = [], []
        for g in genes_raw:
            key = g if g in vocab else norm(g)
            if key in vocab:
                keep_genes.append(norm(g))
                keep_idx.append(vocab[key])
        if len(keep_genes) < 10:
            print(f"[skip] {ds}: too few vocab genes ({len(keep_genes)})")
            continue
        emb = W[np.asarray(keep_idx)]
        print(f"[{ds}] genes in vocab: {len(keep_genes)}")

        # also add GT TFs into tf filter if present
        tf_use = set(tf_set)
        out_tsv = pred_dir / f"RegFormer_tokenemb_{ds}.tsv"
        if args.skip_existing and out_tsv.is_file():
            pred = pd.read_csv(out_tsv, sep="\t")
            print(f"  loaded {out_tsv} edges={len(pred)}")
        else:
            pred = build_edges(keep_genes, emb, tf_use, top_k=args.top_k, min_sim=args.min_sim)
            pred.to_csv(out_tsv, sep="\t", index=False)
            print(f"  wrote {out_tsv} edges={len(pred)}")

        for gt in args.gt_types:
            sub, pat = GT_SOURCES[gt]
            net = args.gt_root / sub / pat.format(dataset=ds)
            if not net.is_file():
                rows.append({"Method": "RegFormer_tokenemb", "Dataset": ds, "GroundTruth": gt, "Status": "MissingGT"})
                continue
            tfs, genes, edges = load_gt(net)
            genes_u = {g for g in genes if g in set(keep_genes)}
            tfs_u = {g for g in tfs if g in set(keep_genes)}
            edges_u = {(a, b) for a, b in edges if a in genes_u and b in genes_u}
            if not tfs_u or not edges_u:
                rows.append({"Method": "RegFormer_tokenemb", "Dataset": ds, "GroundTruth": gt, "Status": "NoOverlap"})
                continue
            metrics = evaluate(pred, tfs_u, genes_u, edges_u)
            rows.append({
                "Method": "RegFormer_tokenemb",
                "Dataset": ds,
                "GroundTruth": gt,
                "Status": "Success",
                "n_genes_vocab": len(keep_genes),
                **metrics,
            })
            print(f"  [{gt}] AUPR_Ratio={metrics['AUPR_Ratio']:.4f} EPR={metrics['EPR']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(args.outdir / "regformer_tokenemb_aupr_epr.csv", index=False)
    ok = df[df.Status == "Success"] if "Status" in df.columns else df
    if not ok.empty:
        wide = ok.pivot_table(index=["Dataset"], columns="GroundTruth", values="AUPR_Ratio", aggfunc="first")
        wide.to_csv(args.outdir / "regformer_tokenemb_AUPR_Ratio_wide.csv")
        print("\n=== AUPR_Ratio ===")
        print(wide)
    print(f"\nDone → {args.outdir}")


if __name__ == "__main__":
    main()
