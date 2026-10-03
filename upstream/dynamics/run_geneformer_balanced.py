

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geneformer swap=8 | 6个数据集 | 输出纯JSON收敛曲线 | 无图无多余输出
"""

import os
import json
import pickle
import warnings
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import BertForMaskedLM

warnings.filterwarnings("ignore")

# =====================================================
# 配置
# =====================================================
MODEL_DIR = "/mnt/10T/yzn/benchmark_GRN/model/weights/Geneformer/default/6L"
DICTS_DIR = "/mnt/10T/yzn/benchmark_GRN/model/weights/Geneformer/dicts"
OUTPUT_JSON = "/mnt/10T/yzn/scGRN-Bench/FBplot/fig4/balanced_convergence_work/geneformer_balanced_accuracy_curves.json"

DATASET_CONFIG = {
    "hESC": {
        "expr_csv": "/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/hESC_chip_matched-ExpressionData.csv",
        "pt_csv": "/mnt/10T/yzn/benchmark_GRN/PseudoTime/hESC/PseudoTime.csv",
        "species": "human",
    },
    "hHep": {
        "expr_csv": "/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/hHep_chip_matched-ExpressionData.csv",
        "pt_csv": "/mnt/10T/yzn/benchmark_GRN/PseudoTime/hHep/PseudoTime.csv",
        "species": "human",
    },
    "mDC": {
        "expr_csv": "/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/mDC_chip_matched-ExpressionData.csv",
        "pt_csv": "/mnt/10T/yzn/benchmark_GRN/PseudoTime/mDC/PseudoTime.csv",
        "species": "mouse",
    },
    "mHSC-E": {
        "expr_csv": "/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/mHSC-E_chip_matched-ExpressionData.csv",
        "pt_csv": "/mnt/10T/yzn/benchmark_GRN/PseudoTime/mHSC-E/PseudoTime.csv",
        "species": "mouse",
    },
    "mHSC-GM": {
        "expr_csv": "/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/mHSC-GM_chip_matched-ExpressionData.csv",
        "pt_csv": "/mnt/10T/yzn/benchmark_GRN/PseudoTime/mHSC-GM/PseudoTime.csv",
        "species": "mouse",
    },
    "mHSC-L": {
        "expr_csv": "/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/mHSC-L_chip_matched-ExpressionData.csv",
        "pt_csv": "/mnt/10T/yzn/benchmark_GRN/PseudoTime/mHSC-L/PseudoTime.csv",
        "species": "mouse",
    },
}

FIXED_PARAMS = {
    "top_percent": 30,
    "pt_quantile": 0.2,
    "gen_iters": 16,
    "max_len": 1024,
    "temperature": 1.0,
    "use_log1p": True,
}
# Align with scFoundation: mapped-only top%; near-zero pred/true are not Up/Down.
EPS_DIR = 1e-3

# =====================================================
# 工具函数
# =====================================================
def normalize_symbol(s):
    return str(s).strip().upper()

def is_ensembl_id(s):
    return isinstance(s, str) and (s.startswith("ENSG") or s.startswith("ENSMUSG"))

def read_pt_file(path):
    try:
        pt_df = pd.read_csv(path, header=None)
        _ = float(pt_df.iloc[0, 1])
    except:
        pt_df = pd.read_csv(path)
    pt_df = pt_df.rename(columns={pt_df.columns[0]: "cell", pt_df.columns[1]: "pt"}).set_index("cell")
    pt_df["pt"] = pd.to_numeric(pt_df["pt"], errors="coerce")
    return pt_df.dropna()

def load_geneformer_dicts(d):
    with open(d/"token_dictionary.pkl", "rb") as f: vocab = pickle.load(f)
    with open(d/"gene_name_id_dict.pkl", "rb") as f: gnd = pickle.load(f)
    return vocab, gnd, int(vocab["<pad>"]), int(vocab["<mask>"])

def build_symbol_to_ensembl(gnd):
    sample = list(gnd.items())[:2000]
    k_ens = sum(is_ensembl_id(str(k)) for k,_ in sample)
    v_ens = sum(is_ensembl_id(str(v)) for _,v in sample)
    if v_ens > k_ens:
        return {normalize_symbol(k):str(v) for k,v in gnd.items()}
    return {normalize_symbol(v):str(k) for k,v in gnd.items()}

def direction_accuracy_top_genes(pred, true, idx, eps=EPS_DIR):
    td = true[idx]
    pd = pred[idx]
    true_dir = np.where(td > eps, 1, np.where(td < -eps, -1, 0))
    pred_dir = np.where(pd > eps, 1, np.where(pd < -eps, -1, 0))
    valid = true_dir != 0
    if not valid.any():
        return float("nan")
    return float((pred_dir[valid] == true_dir[valid]).mean())

def balanced_accuracy_top_genes(pred, true, idx, eps=EPS_DIR):
    td = true[idx]
    pd = pred[idx]
    true_dir = np.where(td > eps, 1, np.where(td < -eps, -1, 0))
    pred_dir = np.where(pd > eps, 1, np.where(pd < -eps, -1, 0))
    valid = true_dir != 0
    recalls = []
    for cls in (1, -1):
        m = valid & (true_dir == cls)
        if m.any():
            recalls.append(float((pred_dir[m] == cls).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")

def positions_dict_from_seq(seq, L):
    pos = {}
    for i in range(L):
        t = int(seq[i])
        if t not in pos: pos[t] = i
    return pos

# =====================================================
# 核心采样
# =====================================================
@torch.no_grad()
def iterative_swap_sampling_geneformer(model, seq, L, pad, mask, device, swap_trials, temp):
    x = seq.clone().to(device).unsqueeze(0)
    attn = torch.zeros((1, x.size(1)), dtype=torch.long, device=device)
    attn[0,:L] = 1
    banned = torch.tensor([pad, mask], device=device)
    for _ in range(swap_trials):
        if L < 2: break
        ij = torch.randperm(L, device="cpu")[:2]
        i,j = int(ij[0]), int(ij[1])
        if i==j: continue
        a,b = int(x[0,i]), int(x[0,j])
        if a in (pad,mask) or b in (pad,mask): continue
        xm = x.clone()
        xm[0,i], xm[0,j] = mask, mask
        logits = model(input_ids=xm, attention_mask=attn).logits[0,[i,j],:].float()
        logits /= max(temp,1e-6)
        bd = banned[(banned>=0)&(banned<logits.size(-1))]
        if bd.numel()>0: logits[:,bd] = -1e9
        logp = torch.log_softmax(logits,-1)
        if a<0 or b<0 or a>=logits.size(-1) or b>=logits.size(-1): continue
        cur = logp[0,a].item() + logp[1,b].item()
        swp = logp[0,b].item() + logp[1,a].item()
        if swp > cur:
            x[0,i], x[0,j] = x[0,j].clone(), x[0,i].clone()
    return x.squeeze(0).cpu().numpy()

# =====================================================
# 运行单个数据集，返回16步accuracy列表
# =====================================================
@torch.no_grad()
def run_dataset(dset_name, dset_cfg, model, vocab, gnd, pad, mask, device):
    print(f"\n=== Running {dset_name} (swap=8) ===")
    expr = pd.read_csv(dset_cfg["expr_csv"], index_col=0)
    pt_df = read_pt_file(dset_cfg["pt_csv"])
    common = expr.columns.intersection(pt_df.index)
    expr = expr[common]
    pt = pt_df.loc[common,"pt"].values

    X = expr.T.values.astype(np.float32)
    if FIXED_PARAMS["use_log1p"]: X = np.log1p(np.maximum(X,0))

    lo, hi = np.quantile(pt, [0.2, 0.8])
    early_mask = pt <= lo
    late_mask = pt >= hi

    early_mean = X[early_mask].mean(0)
    late_mean = X[late_mask].mean(0)
    true_delta = late_mean - early_mean

    sym2ens = build_symbol_to_ensembl(gnd)
    genes = expr.index.astype(str).tolist()

    def get_tid(g):
        gn = normalize_symbol(g)
        if gn in vocab: return vocab[gn]
        ens = sym2ens.get(gn)
        if ens and ens in vocab: return vocab[ens]
        return pad
    gene_tids = np.array([get_tid(g) for g in genes], dtype=np.int64)
    mapped_pool = np.where(gene_tids != pad)[0]
    if len(mapped_pool) == 0:
        raise ValueError(f"{dset_name}: no genes mapped to Geneformer vocab")
    top_n = max(int(len(mapped_pool) * FIXED_PARAMS["top_percent"] / 100), 1)
    order = np.argsort(-np.abs(true_delta[mapped_pool]))
    top_idx = mapped_pool[order[:top_n]]
    print(
        f"  mapped={len(mapped_pool)}/{len(genes)}; "
        f"eval top{FIXED_PARAMS['top_percent']}% mapped={top_n}; EPS_DIR={EPS_DIR}"
    )

    n_genes = len(true_delta)

    def cell2seq(x):
        order = np.argsort(-x)
        seq = []
        for j in order:
            t = gene_tids[j]
            if t==pad: continue
            seq.append(int(t))
            if len(seq)>=1024: break
        L = len(seq)
        if L<1024: seq += [pad]*(1024-L)
        return seq, L

    early_cells = []
    for i in np.where(early_mask)[0]:
        seq, L = cell2seq(X[i])
        if L>10: early_cells.append( (seq,L) )

    init = [np.array(s,dtype=np.int64) for s,_ in early_cells]
    curr = [x.copy() for x in init]
    lengths = [L for _,L in early_cells]

    acc_curve = []
    ba_curve = []

    for it in range(16):
        deltas = []
        next_curr = []

        for ci, L in enumerate(lengths):
            x0 = init[ci]
            xp = curr[ci]
            x1 = iterative_swap_sampling_geneformer(
                model, torch.tensor(xp,dtype=torch.long), L, pad, mask, device,
                swap_trials=8, temp=1.0
            )
            next_curr.append(x1)
            p0 = positions_dict_from_seq(x0, L)
            p1 = positions_dict_from_seq(x1, L)
            d = np.zeros(n_genes, dtype=np.float32)
            for gi in mapped_pool:
                t = int(gene_tids[gi])
                d[gi] = p1.get(t,L) - p0.get(t,L)
            deltas.append(d)

        # Accept every refinement step (paper Algorithm; no accuracy-based rollback).
        curr = next_curr
        mean_d = np.stack(deltas).mean(0)
        pred = -mean_d
        acc = direction_accuracy_top_genes(pred, true_delta, top_idx)
        ba = balanced_accuracy_top_genes(pred, true_delta, top_idx)
        acc_curve.append(acc)
        ba_curve.append(ba)
        print(f"  Iter {it+1:2d}; acc = {acc:.6f}; balanced_acc = {ba:.6f}")

    return ba_curve

# =====================================================
# 主函数：跑6个数据集，输出JSON
# =====================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | swap=8 | 6 datasets | output JSON only")

    vocab, gnd, pad, mask = load_geneformer_dicts(Path(DICTS_DIR))
    model = BertForMaskedLM.from_pretrained(MODEL_DIR).to(device).eval()

    results = {}
    for dset, cfg in DATASET_CONFIG.items():
        try:
            curve = run_dataset(dset, cfg, model, vocab, gnd, pad, mask, device)
            results[dset] = curve
        except Exception as e:
            print(f"Failed {dset}: {e}")
            results[dset] = []

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Done! JSON saved to: {OUTPUT_JSON}")
    print("\nPreview:")
    for k,v in results.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()