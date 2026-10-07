#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scPRINT 伪时间方向基准（与 `run_unified_multidataset_pseudotime.py` 任务对齐的独立入口）

## scPRINT 如何做「类似」任务（与 scGPT / Geneformer 的差异）

- **Geneformer / LangCell**：离散基因 token + MLM / swap，在序列空间迭代改写 token，用 rank 或隐式表达变化。
- **scGPT**：连续 binned expression + 基因顺序固定，每步用 `mlm_output` EMA 更新，再与 early 平均求 `pred_delta`。
- **scPRINT**（见 `scprint/tasks/denoise.py`、`scprint/model/model.py::_predict`）：
  - 输入是 **连续表达** + `scdataloader.Collator` 选出的基因子集（最多 `max_len`）。
  - 推理主路径是 **`predict_mode="denoise"`**：`forward(..., depth_mult=expression.sum(1)*depth_mult, req_depth=depth*depth_mult)`，
    得到 ZINB 头的 **`mean`**（及可选 disp / zero），这是 **去噪/重构式** 预测，不是按伪时间「生成晚期状态」的显式目标。
  - 官方 `Denoiser` 走完整 DataLoader，并把结果写入 `h5ad`；本脚本与 **`my_GRN_code/infer_raw_attn_to_tsv.py`** 一致：
    **`torch.load` + `inspect.signature` 过滤 `hyper_parameters`**（避免 ckpt 里 `memmap_gene_emb` 等字段触发 Lightning/jsonargparse 报错）、
    可选 **`--token-pkl` 对齐 gene embedding**、**`disable_bias_runtime` forward patch**；
    迭代阶段使用与 **`infer_grn_fp32_noamp_slice_attn`** 相同的 **`gene_pos` 张量 + 按 batch `_predict(denoise)`**（默认 **fp32 无 AMP**），对早期细胞做 EMA 更新后再算方向准确率。

依赖（与官方 README 一致）：`scprint`、`scdataloader`、`lamin`/`lamindb`、`scanpy`、`anndata` 等；首次运行通常需要
`scdataloader` 的 ontology：`populate_my_ontology(...)`。

用法示例：

  python run_scprint_multidataset_pseudotime.py \\
    --checkpoint /path/to/model.ckpt \\
    --scprint-repo /mnt/10T/yzn/scPRINT \\
    --outdir pre_scprint_results \\
    --save-trajectory-plot --trajectory-embed umap
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import pickle
import re
import sys
from pathlib import Path
from types import MethodType
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
try:
    from .direction_metrics import (EPS_DIR, balanced_direction_accuracy,
                                    direction_signs, metric_metadata, save_accuracy_curves)
except ImportError:  # Direct script execution.
    from direction_metrics import (EPS_DIR, balanced_direction_accuracy,
                                   direction_signs, metric_metadata, save_accuracy_curves)

from scipy.sparse import issparse

# ---------------------------------------------------------------------------
# 与 unified 脚本一致的数据集定义（避免 import unified 牵连 Geneformer/scGPT）
# ---------------------------------------------------------------------------

DEFAULT_DATASETS: Dict[str, dict] = {
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


def read_pt_file(path: str) -> pd.DataFrame:
    try:
        pt_df = pd.read_csv(path, header=None)
        _ = float(pt_df.iloc[0, 1])
    except (ValueError, IndexError, TypeError):
        pt_df = pd.read_csv(path)
    if pt_df.shape[1] < 2:
        raise ValueError(f"Invalid pseudotime file: {path}")
    pt_df = pt_df.rename(columns={pt_df.columns[0]: "cell", pt_df.columns[1]: "pt"}).set_index("cell")
    pt_df["pt"] = pd.to_numeric(pt_df["pt"], errors="coerce")
    pt_df = pt_df.dropna(subset=["pt"])
    return pt_df


def organism_id(species: str) -> str:
    return "NCBITaxon:10090" if species == "mouse" else "NCBITaxon:9606"


def direction_accuracy_top_genes(
    pred_delta: np.ndarray,
    true_delta: np.ndarray,
    top_idx: np.ndarray,
    eps: float = EPS_DIR,
) -> Tuple[float, float]:
    """Return balanced direction accuracy and inverted-truth BA (legacy API name).

    Near-zero truth is excluded; near-zero prediction is incorrect. Even when
    eps=0, exact zero is undefined and is never assigned to Down.
    """
    return balanced_direction_accuracy(pred_delta, true_delta, top_idx, eps)


def _pca_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    return (u[:, :2] * s[:2]).astype(np.float32)


def save_trajectory_plot(
    ds_dir: Path,
    name: str,
    pred_delta_by_iter: np.ndarray,
    true_delta: np.ndarray,
    save_plot: bool,
    embed: str,
    seed: int,
) -> None:
    if not save_plot:
        return
    import matplotlib.pyplot as plt

    traj = np.asarray(pred_delta_by_iter, dtype=np.float32)
    td = np.asarray(true_delta, dtype=np.float32).ravel()
    if traj.size == 0 or td.shape[0] != traj.shape[1]:
        return

    np.save(ds_dir / "trajectory_raw_matrix.npy", traj)
    traj_c = np.nan_to_num(traj, nan=0.0, posinf=0.0, neginf=0.0)
    all_X = np.vstack(
        [np.zeros((1, traj.shape[1]), dtype=np.float32), traj_c, np.nan_to_num(td.reshape(1, -1), nan=0.0)]
    )
    np.save(ds_dir / "trajectory_embed_input_matrix.npy", all_X)

    n_traj = int(traj.shape[0])
    if embed == "umap":
        try:
            import umap
        except Exception as e:
            raise RuntimeError("UMAP 需要: pip install umap-learn") from e
        nn = min(15, max(2, all_X.shape[0] - 1))
        coords = umap.UMAP(
            n_components=2, n_neighbors=nn, min_dist=0.15, metric="euclidean", random_state=seed
        ).fit_transform(all_X).astype(np.float32)
    else:
        coords = _pca_2d(all_X)

    emb = embed
    df = pd.DataFrame(
        {
            "point": ["start_delta0"] + [f"iter_{k + 1}" for k in range(n_traj)] + ["true_delta_target"],
            "iter": np.array([0] + list(range(1, n_traj + 1)) + [-1], dtype=np.int64),
            f"{emb}1": coords[:, 0],
            f"{emb}2": coords[:, 1],
        }
    )
    df.to_csv(ds_dir / f"trajectory_{emb}2d.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=200)
    sl = slice(1, n_traj + 1)
    c_traj = coords[sl]
    ax.plot(c_traj[:, 0], c_traj[:, 1], "-", color="0.7", lw=1.1, zorder=1)
    sca = ax.scatter(
        c_traj[:, 0], c_traj[:, 1], c=np.arange(1, n_traj + 1), cmap="viridis", s=45, zorder=2, edgecolors="white", linewidths=0.35
    )
    fig.colorbar(sca, ax=ax, shrink=0.72, label="Iteration")
    i0, i_pred, i_true = 0, n_traj, n_traj + 1
    ax.scatter(coords[i0, 0], coords[i0, 1], s=160, c="#1b9e77", edgecolors="black", linewidths=0.7, zorder=5, label="Start (Δ=0)")
    ax.scatter(coords[i_pred, 0], coords[i_pred, 1], s=160, c="#d95f02", edgecolors="black", linewidths=0.7, zorder=5, label="Pred end (last iter)")
    ax.scatter(coords[i_true, 0], coords[i_true, 1], s=160, c="#7570b3", edgecolors="black", linewidths=0.7, zorder=5, label="True Δ (late−early)")
    for a, b, col in [(i0, i_pred, "0.35"), (i_pred, i_true, "0.45")]:
        ax.annotate(
            "",
            xy=(coords[b, 0], coords[b, 1]),
            xytext=(coords[a, 0], coords[a, 1]),
            arrowprops=dict(arrowstyle="->", color=col, lw=1.4, shrinkA=10, shrinkB=10),
            zorder=3,
        )
    ax.set_xlabel(f"{embed.upper()}1")
    ax.set_ylabel(f"{embed.upper()}2")
    ax.set_title(f"{name}\npred_delta trajectory (scPRINT denoise-EMA); 2D={embed.upper()}", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(ds_dir / f"trajectory_{emb}2d.png", dpi=300)
    plt.close(fig)


def _fix_genes_in_ckpt_dict(ckpt: dict) -> dict:
    hp = ckpt.get("hyper_parameters") or {}
    g = hp.get("genes")
    if isinstance(g, dict):
        if "gene_list" in g:
            genes_list = g["gene_list"]
        elif "genes" in g:
            genes_list = g["genes"]
        elif "names" in g:
            genes_list = g["names"]
        else:
            vals = list(g.values())
            genes_list = vals[0] if vals and isinstance(vals[0], list) else list(g.keys())
        hp["genes"] = list(genes_list)
    ckpt["hyper_parameters"] = hp
    return ckpt


def _normalize_state_dict_keys(state_dict: dict) -> dict:
    prefixes = ["model.", "scprint.", "net.", "module."]
    out = {}
    for k, v in state_dict.items():
        kk = k
        for p in prefixes:
            if kk.startswith(p):
                kk = kk[len(p) :]
                break
        out[kk] = v
    return out


def _clean_symbol(sym: str) -> str:
    s = str(sym).strip()
    s = re.sub(r"\s+", "", s)
    return s


def load_env_genes_from_token_pkl(token_pkl: str) -> Tuple[List[str], List[Optional[str]]]:
    """token_dictionary.pkl: dict(token->id)，与 my_GRN_code/infer_raw_attn_to_tsv.py 一致。"""
    with open(token_pkl, "rb") as f:
        stoi = pickle.load(f)
    if not isinstance(stoi, dict):
        raise TypeError(f"Expected dict token->id, got {type(stoi)}")
    max_id = max(stoi.values())
    itos: List[Optional[str]] = [None] * (max_id + 1)
    for tok, idx in stoi.items():
        itos[idx] = str(tok)
    if any(x is None for x in itos):
        raise RuntimeError("token id 不连续或缺失")
    if len(itos) >= 2 and itos[0] == "<pad>" and itos[1] == "<mask>":
        genes = itos[2:]
    else:
        special = {"<pad>", "<mask>", "<unk>", "<cls>", "<bos>", "<eos>"}
        genes = [t for t in itos if isinstance(t, str) and t not in special]
    genes = [g for g in genes if g is not None]
    return genes, itos


def _model_genes_mostly_human_ensembl(model_genes: Set[str], max_check: int = 4000) -> bool:
    lst = [str(g) for g in list(model_genes)[:max_check]]
    if not lst:
        return False
    return sum(1 for g in lst if g.startswith("ENSG")) / len(lst) >= 0.55


def _mygene_taxon_for_symbol_map(dataset_taxon: str, model_genes: Set[str]) -> str:
    """
    小鼠矩阵 + 人类参考模型（如 *9606*.ckpt、基因多为 ENSG）时，若用 mouse 查 mygene 会得到 ENSMUSG，
    与模型基因集无交集。此时改用 human 查 symbol→ENSG（许多 symbol 与人同源一致），
    与 my_GRN_code/infer_raw_attn_to_tsv.py 里对鼠数据仍写 SPECIES=NCBITaxon:9606 的做法一致。
    """
    if dataset_taxon == "NCBITaxon:10090" and _model_genes_mostly_human_ensembl(model_genes):
        return "NCBITaxon:9606"
    return dataset_taxon


def map_symbols_to_ensembl(symbols: List[str], species_taxon: str, target_ens_set: Set[str]) -> Dict[str, str]:
    try:
        from mygene import MyGeneInfo
    except Exception as e:
        raise RuntimeError("需要 mygene: pip install mygene") from e
    mg = MyGeneInfo()
    sp_name = "human" if species_taxon == "NCBITaxon:9606" else "mouse"
    res = mg.querymany(
        [_clean_symbol(s) for s in symbols],
        scopes="symbol",
        fields="ensembl.gene",
        species=sp_name,
        as_dataframe=True,
        returnall=False,
        verbose=False,
    )
    mapping: Dict[str, str] = {}
    for sym, row in res.iterrows():
        val = row.get("ensembl.gene", None)
        ens = None
        if isinstance(val, dict) and "gene" in val:
            ens = val["gene"]
        elif isinstance(val, list) and len(val) > 0:
            v0 = val[0]
            if isinstance(v0, dict) and "gene" in v0:
                ens = v0["gene"]
            elif isinstance(v0, str):
                ens = v0
        elif isinstance(val, str):
            ens = val
        if ens is not None and ens in target_ens_set:
            mapping[str(sym)] = ens
    return mapping


def disable_bias_runtime(model: torch.nn.Module) -> None:
    """与 infer_raw_attn_to_tsv.py 中 disable_bias_runtime 一致：attn_bias none + forward patch。"""
    model.attn_bias = "none"
    if hasattr(model, "nbias"):
        try:
            delattr(model, "nbias")
        except Exception:
            pass

    def new_forward(
        self,
        gene_pos,
        expression=None,
        mask=None,
        req_depth=None,
        timepoint=None,
        get_gene_emb=False,
        metacell_token=None,
        depth_mult=None,
        do_sample=False,
        do_mvc=False,
        do_class=False,
        get_attention_layer=None,
    ):
        if get_attention_layer is None:
            get_attention_layer = []
        encoding = self._encoder(
            gene_pos,
            expression,
            mask,
            req_depth=req_depth if self.depth_atinput else None,
            timepoint=timepoint,
            metacell_token=metacell_token,
        )
        if self.cell_transformer:
            cell_encoding = encoding[:, : self.cell_embs_count, :]
            encoding = encoding[:, self.cell_embs_count :, :]
        transformer_output = self.transformer(
            encoding,
            return_qkv=get_attention_layer,
            bias=None,
            bias_layer=list(range(self.nlayers - 1)),
        )
        if len(get_attention_layer) > 0:
            transformer_output, qkvs = transformer_output
        if self.cell_transformer:
            cell_output = self.cell_transformer(cell_encoding, x_kv=transformer_output)
            transformer_output = torch.cat([cell_output, transformer_output], dim=1)
        depth_mult2 = expression.sum(1) if depth_mult is None else depth_mult
        res = self._decoder(
            transformer_output,
            depth_mult2,
            get_gene_emb,
            do_sample,
            do_mvc,
            do_class,
            req_depth=req_depth if not self.depth_atinput else None,
        )
        return (res, qkvs) if len(get_attention_layer) > 0 else res

    model.forward = MethodType(new_forward, model)


def load_scprint_model(
    checkpoint: Path,
    scprint_repo: Path,
    transformer: str,
    fix_genes_ckpt: bool,
    token_pkl: Optional[str],
    gene_emb_offset: int,
    patch_disable_bias: bool,
) -> torch.nn.Module:
    """
    手动 torch.load + inspect.signature 过滤 hyper_parameters，避免 Lightning/jsonargparse 对 memmap_gene_emb 等字段报错。
    若提供 token_pkl，则与 infer_raw_attn_to_tsv.py 相同：按 pkl 基因表重排/对齐 embedding 再 load_state_dict。
    """
    if str(scprint_repo) not in sys.path:
        sys.path.insert(0, str(scprint_repo))
    from scprint import scPrint

    ck = torch.load(checkpoint, map_location="cpu")
    if fix_genes_ckpt:
        ck = _fix_genes_in_ckpt_dict(ck)
    hp_block = ck.get("hyper_parameters")
    if hp_block is None:
        raise RuntimeError("checkpoint 缺少 hyper_parameters")
    hp = dict(hp_block)
    state = ck.get("state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint 缺少 state_dict")
    state = _normalize_state_dict_keys(state)

    sig = inspect.signature(scPrint.__init__)
    param_names = {
        n
        for n, p in sig.parameters.items()
        if n != "self" and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    LIGHTNING_NOISE = {"_target_", "_recursive_", "class_path", "init_args"}
    kwargs = {k: v for k, v in hp.items() if k in param_names and k not in LIGHTNING_NOISE}
    for drop_k in ("nb_features", "memmap_gene_emb"):
        kwargs.pop(drop_k, None)

    kwargs["transformer"] = transformer
    kwargs["precpt_gene_emb"] = None
    if patch_disable_bias and "attn_bias" in param_names:
        kwargs["attn_bias"] = "none"

    ckpt_genes = hp.get("genes", None)
    if not isinstance(ckpt_genes, (list, tuple)) or len(ckpt_genes) == 0:
        raise RuntimeError("hyper_parameters 中 genes 无效")

    emb_key = "gene_encoder.embeddings.weight"
    ckpt_emb = state.get(emb_key, None)
    if not torch.is_tensor(ckpt_emb):
        raise RuntimeError(f"state_dict 缺少 {emb_key}")

    if token_pkl:
        env_genes, _ = load_env_genes_from_token_pkl(token_pkl)
        kwargs.pop("genes", None)
        kwargs["genes"] = env_genes
    else:
        if "genes" not in kwargs:
            kwargs["genes"] = list(ckpt_genes)

    model = scPrint(**kwargs)
    model = model.to(torch.float32).eval()
    model_sd = model.state_dict()
    mw = model_sd.get(emb_key, None)
    if not torch.is_tensor(mw):
        raise RuntimeError(f"模型内缺少 {emb_key}")

    if mw.shape[1] != ckpt_emb.shape[1]:
        raise RuntimeError(f"embedding 维度不一致 ckpt={tuple(ckpt_emb.shape)} model={tuple(mw.shape)}")

    if token_pkl:
        ckpt_idx = {str(g): i for i, g in enumerate(ckpt_genes)}
        model_genes = [str(g) for g in model.genes]
        new_emb = mw.clone()
        copied = 0
        for j, g in enumerate(model_genes):
            src = ckpt_idx.get(g, -1)
            if src >= 0:
                src2 = src + int(gene_emb_offset)
                if 0 <= src2 < ckpt_emb.shape[0]:
                    new_emb[j] = ckpt_emb[src2]
                    copied += 1
        state = dict(state)
        state[emb_key] = new_emb
        print(f"[scPRINT] token_pkl 对齐: 复制 embedding 行 {copied}/{len(model_genes)}")

    filtered = {}
    skipped = []
    for k, v in state.items():
        if k not in model_sd:
            continue
        if not torch.is_tensor(v):
            continue
        if tuple(v.shape) == tuple(model_sd[k].shape):
            filtered[k] = v
        else:
            skipped.append((k, tuple(v.shape), tuple(model_sd[k].shape)))

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"[scPRINT] load_state_dict keys={len(filtered)} skipped_shape={len(skipped)} missing={len(missing)}")
    if patch_disable_bias:
        disable_bias_runtime(model)
    return model


def expr_to_ensembl_rows(expr: pd.DataFrame, genes_syms: List[str], model_genes: Set[str], species_taxon: str) -> pd.DataFrame:
    """将基因行 symbol -> Ensembl，仅保留落在 model_genes 的行，同名 ENSG 合并。"""
    query_taxon = _mygene_taxon_for_symbol_map(species_taxon, model_genes)
    if query_taxon != species_taxon:
        print(
            f"  [scPRINT] 数据 taxon={species_taxon} 但模型多为人类 ENSG，"
            f"mygene 使用 taxon={query_taxon} 做 symbol→Ensembl（鼠数据×人模型）。"
        )
    mapping = map_symbols_to_ensembl(genes_syms, query_taxon, model_genes)
    if len(mapping) == 0:
        raise RuntimeError(
            "symbol->Ensembl 后与模型基因无交集。请检查："
            "1) 矩阵行名是否为 gene symbol；2) --token-pkl 与 ckpt 是否配套；"
            "3) 鼠数据+人模型时若仍失败，可能需换用小鼠 ckpt 或自行做同源映射。"
        )
    df = expr.loc[[k for k in mapping.keys() if k in expr.index]].copy()
    df.index = [mapping[str(i)] for i in df.index]
    df = df.groupby(df.index).sum()
    return df


def pick_eval_genes(
    expr: pd.DataFrame,
    genes_row_order: List[str],
    model_genes: Sequence[str],
    max_len: int,
) -> List[str]:
    mg = set(model_genes)
    cand = [g for g in genes_row_order if g in mg]
    if len(cand) <= max_len:
        return cand
    sub = expr.loc[cand]
    var = sub.var(axis=1).sort_values(ascending=False)
    return var.index[:max_len].astype(str).tolist()


def run_one_dataset(
    name: str,
    cfg: dict,
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[List[float], dict, dict]:
    import scanpy as sc
    from anndata import AnnData

    expr_raw = pd.read_csv(cfg["expr_csv"], index_col=0)
    pt_df = read_pt_file(cfg["pt_csv"])
    genes_original = expr_raw.index.astype(str).tolist()
    sym_index = [g.upper() if cfg.get("species") == "mouse" else str(g) for g in genes_original]
    expr_sym = expr_raw.copy()
    expr_sym.index = sym_index
    expr_sym = expr_sym.groupby(expr_sym.index).sum()
    expr_sym = expr_sym.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    taxon = organism_id(cfg.get("species", "human"))
    if (args.token_pkl or "").strip():
        expr_work = expr_to_ensembl_rows(expr_sym, list(expr_sym.index), set(model.genes), taxon)
        panel_ids = expr_work.index.astype(str).tolist()
    else:
        expr_work = expr_sym
        panel_ids = expr_work.index.astype(str).tolist()

    common = expr_work.columns.intersection(pt_df.index)
    if len(common) == 0:
        raise ValueError("表达矩阵与伪时间无重叠细胞")
    expr_work = expr_work[common]
    pt = pt_df.loc[common, "pt"].to_numpy(dtype=np.float64)

    lo, hi = np.quantile(pt, [args.pt_quantile, 1 - args.pt_quantile])
    early_m = pt <= lo
    late_m = pt >= hi
    if early_m.sum() == 0 or late_m.sum() == 0:
        raise ValueError("分位数切分后缺少 early / late 细胞")

    eval_genes = pick_eval_genes(expr_work, panel_ids, model.genes, args.max_len)
    if len(eval_genes) < 2:
        raise ValueError("模型可映射的基因过少")

    missing_g = [g for g in eval_genes if g not in expr_work.index]
    if missing_g:
        raise RuntimeError(f"eval_genes 不在表达矩阵索引中: {missing_g[:5]}")

    X_sub = expr_work.loc[eval_genes].T.to_numpy(dtype=np.float32)
    cell_ids = expr_work.columns.astype(str).tolist()

    ad_all = AnnData(
        X=np.asarray(X_sub, dtype=np.float32),
        obs=pd.DataFrame(index=cell_ids),
        var=pd.DataFrame(index=eval_genes),
    )
    ad_all.obs["organism_ontology_term_id"] = taxon
    if args.normalize_target_sum is not None and args.normalize_target_sum > 0:
        sc.pp.normalize_total(ad_all, target_sum=args.normalize_target_sum)
    if args.log1p:
        sc.pp.log1p(ad_all)

    early_mean = np.asarray(ad_all.X[early_m].mean(axis=0), dtype=np.float32).ravel()
    late_mean = np.asarray(ad_all.X[late_m].mean(axis=0), dtype=np.float32).ravel()
    true_delta = late_mean - early_mean
    n_sub = len(eval_genes)
    top_n = max(int(n_sub * args.top_percent / 100), 1)
    top_local = np.argsort(np.abs(true_delta))[::-1][:top_n]

    model = model.to(device).eval()
    model.pred_log_adata = False
    model.doplot = False
    if not args.use_amp:
        model = model.float()

    gene_to_idx = {str(g): i for i, g in enumerate(model.genes)}
    for g in eval_genes:
        if str(g) not in gene_to_idx:
            raise RuntimeError(f"基因不在 model.genes: {g}")
    gene_pos_1 = torch.tensor([gene_to_idx[str(g)] for g in eval_genes], dtype=torch.long, device=device)

    xm = ad_all[early_m].X
    # NumPy 1.x 不支持 np.asarray(..., copy=True)（仅 2.x）；用 np.array(..., copy=True)
    if issparse(xm):
        state = np.asarray(xm.toarray(), dtype=np.float32)
    else:
        state = np.array(np.asarray(xm), dtype=np.float32, copy=True)
    n_early = int(state.shape[0])
    bs = int(args.batch_size)

    acc_curve: List[float] = []
    acc_inv: List[float] = []
    pred_delta_by_iter: List[np.ndarray] = []
    state_by_iter: List[np.ndarray] = []  # [gen_iters, n_early, n_eval_genes] in current state space

    amp_dtype = torch.float16 if device.type == "cuda" else torch.float32

    def postprocess_pred_expr_like_state(mu: torch.Tensor) -> torch.Tensor:
        """
        Align scPRINT denoise output to the same space as `state`.
        Current `state` is built from ad_all after optional normalize_total + log1p.
        """
        x = mu.float().clamp_min(0.0)
        if args.normalize_target_sum is not None and args.normalize_target_sum > 0:
            lib = x.sum(dim=1, keepdim=True).clamp_min(1e-6)
            x = x / lib * float(args.normalize_target_sum)
        if args.log1p:
            x = torch.log1p(x)
        return x

    for it in range(args.gen_iters):
        model.on_predict_epoch_start()
        with torch.no_grad():
            for start in range(0, n_early, bs):
                end = min(start + bs, n_early)
                expr_b = torch.from_numpy(state[start:end]).to(device=device, dtype=torch.float32)
                depth_b = expr_b.sum(dim=1).clamp_min(1e-6)
                gene_pos_b = gene_pos_1.unsqueeze(0).expand(end - start, -1)

                def run_predict(inp: torch.Tensor, dep: torch.Tensor) -> torch.Tensor:
                    if args.use_amp and device.type == "cuda":
                        with torch.autocast(device_type="cuda", dtype=amp_dtype):
                            out = model._predict(
                                gene_pos_b,
                                inp,
                                dep,
                                predict_mode="denoise",
                                pred_embedding=[],
                                get_attention_layer=[],
                                depth_mult=args.predict_depth_mult,
                                keep_output=False,
                            )
                    else:
                        out = model._predict(
                            gene_pos_b,
                            inp,
                            dep,
                            predict_mode="denoise",
                            pred_embedding=[],
                            get_attention_layer=[],
                            depth_mult=args.predict_depth_mult,
                            keep_output=False,
                        )
                    return out["expr"][0].float()

                mu = run_predict(expr_b, depth_b)
                mu = postprocess_pred_expr_like_state(mu)
                new_x = args.ema_alpha * expr_b + (1.0 - args.ema_alpha) * mu
                state[start:end] = new_x.cpu().numpy()

        pred_mean = state.mean(axis=0).astype(np.float32)
        pred_delta = pred_mean - early_mean
        pred_delta_by_iter.append(pred_delta.copy())

        if args.save_cell_preds_by_iter:
            state_by_iter.append(state.astype(np.float32, copy=False).copy())

        acc_now, inv_now = direction_accuracy_top_genes(
            pred_delta,
            true_delta,
            top_local,
            eps=float(getattr(args, "acc_eps", EPS_DIR)),
        )
        acc_curve.append(acc_now)
        acc_inv.append(inv_now)
        if args.print_every > 0 and (it + 1) % args.print_every == 0:
            pred_pos = float((pred_delta > 0).mean())
            true_pos = float((true_delta > 0).mean())
            print(
                f"    Iter {it+1:>2}/{args.gen_iters} | BA={acc_curve[-1]:.2%} | "
                f"inv={acc_inv[-1]:.2%} | pred_pos={pred_pos:.2%} | true_pos={true_pos:.2%}"
            )

    diagnostics = {
        **metric_metadata(float(args.acc_eps)),
        "dataset": name,
        "model": "scprint",
        "n_genes_total": len(panel_ids),
        "n_genes_eval_panel": n_sub,
        "n_cells": int(len(common)),
        "n_early": int(early_m.sum()),
        "n_late": int(late_m.sum()),
        "eval_percent": args.top_percent,
        "eval_genes_count": top_n,
        "final_acc_inv_truth": float(acc_inv[-1]) if acc_inv else None,
    }
    artifacts = {
        "genes": eval_genes,
        "true_delta": true_delta,
        "top_idx": top_local.astype(np.int64),
        "pred_delta_by_iter": np.stack(pred_delta_by_iter, axis=0),
        "preds_full_by_iter": np.stack(state_by_iter, axis=0) if args.save_cell_preds_by_iter and state_by_iter else None,
    }
    return acc_curve, diagnostics, artifacts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="scPRINT 伪时间方向基准（denoise + EMA 迭代）")
    p.add_argument("--checkpoint", type=str, required=True, help="Lightning ckpt 路径（scPrint）")
    p.add_argument("--scprint-repo", type=str, default="/mnt/10T/yzn/scPRINT", help="scPRINT 源码根目录（加入 sys.path）")
    p.add_argument("--outdir", type=str, default="pre_scprint_results_unified")
    p.add_argument("--datasets", type=str, default="", help="逗号分隔子集，默认全部")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--save-cell-preds-by-iter",
        action="store_true",
        default=False,
        help="Save per-iteration per-early-cell denoise+EMA state (disk-heavy).",
    )
    p.add_argument("--pt-quantile", type=float, default=0.2)
    p.add_argument("--top-percent", type=int, default=30)
    p.add_argument("--gen-iters", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-len", type=int, default=4000, help="评估面板最大基因数（与 infer / Denoiser 类似）")
    p.add_argument("--predict-depth-mult", type=float, default=4.0, help="对应 Denoiser.predict_depth_mult")
    p.add_argument(
        "--ema-alpha",
        type=float,
        default=0.9,
        help="EMA retention on previous state (paper α): x <- alpha*x + (1-alpha)*x_hat.",
    )
    p.add_argument("--normalize-target-sum", type=float, default=1e4, help="scanpy normalize_total；<=0 跳过")
    p.add_argument("--log1p", action="store_true", default=True)
    p.add_argument("--no-log1p", action="store_true", default=False)
    p.add_argument("--transformer", type=str, default="normal", choices=["normal", "flash", "performer"], help="加载 ckpt 时传入 scPrint；无 Flash 时用 normal")
    p.add_argument(
        "--token-pkl",
        type=str,
        default="",
        help="Geneformer 式 token_dictionary.pkl：与 infer_raw_attn_to_tsv 一致，按 pkl 基因顺序对齐 embedding；并提供 symbol->Ensembl 评估路径",
    )
    p.add_argument("--gene-emb-offset", type=int, default=0, help="对齐 ckpt embedding 行时的偏移（与 CKPT_GENE_EMB_OFFSET 相同）")
    p.add_argument(
        "--no-patch-disable-bias",
        action="store_true",
        default=False,
        help="默认会 patch forward 去掉 attn bias（与 GRN 推断脚本一致）；若与某 ckpt 不兼容可加此项",
    )
    p.add_argument("--use-amp", action="store_true", default=False, help="GPU 上使用 autocast（默认 fp32，与 infer_grn_fp32_noamp_slice_attn 一致）")
    p.add_argument("--fix-genes-ckpt", action="store_true", default=False, help="torch.load 修复 hyper_parameters['genes'] 为列表再加载（部分 ckpt 需要）")
    p.add_argument("--populate-ontology", action="store_true", default=True, help="首次运行建议开启（scdataloader）")
    p.add_argument("--no-populate-ontology", action="store_true", default=False)
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument(
        "--acc-eps",
        type=float,
        default=EPS_DIR,
        help="Balanced accuracy：默认 eps=1e-3；近零真实变化排除，近零预测计错；eps=0 时精确零仍不属于 Down。",
    )
    p.add_argument("--save-trajectory-plot", action="store_true", default=False)
    p.add_argument("--trajectory-embed", choices=["pca", "umap"], default="umap")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.log1p = args.log1p and (not args.no_log1p)
    if args.no_populate_ontology:
        args.populate_ontology = False

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.populate_ontology:
        try:
            from scdataloader.utils import populate_my_ontology

            populate_my_ontology(
                organisms_clade=["vertebrates"],
                sex=["PATO:0000384", "PATO:0000383"],
            )
        except Exception as e:
            print(f"[WARN] populate_my_ontology 失败（若已 populate 可忽略）: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = Path(args.checkpoint).resolve()
    repo = Path(args.scprint_repo).resolve()
    print(f"Device: {device} | ckpt: {ck}")
    token_pkl = args.token_pkl.strip() or None
    model = load_scprint_model(
        ck,
        repo,
        args.transformer,
        args.fix_genes_ckpt,
        token_pkl,
        args.gene_emb_offset,
        patch_disable_bias=not args.no_patch_disable_bias,
    )
    if not getattr(model, "classes", None):
        print(
            "[WARN] model.classes 为空：若 _predict 报错，请换用带 classifier 的 ckpt，"
            "或向 cantinilab/scPRINT 提 issue（keep_output=False 依赖 pred_embedding 索引）。"
        )

    out_root = Path(args.outdir) / "scprint"
    out_root.mkdir(parents=True, exist_ok=True)

    selected = None
    if args.datasets.strip():
        selected = {x.strip() for x in args.datasets.split(",") if x.strip()}
        bad = sorted(selected - set(DEFAULT_DATASETS.keys()))
        if bad:
            raise ValueError(f"未知数据集: {bad}")

    all_curves: Dict[str, List[float]] = {}
    all_diag: Dict[str, dict] = {}
    errors: Dict[str, str] = {}

    for name, cfg in DEFAULT_DATASETS.items():
        if selected is not None and name not in selected:
            continue
        print(f"\n===== {name} (scPRINT) =====")
        try:
            acc, diag, art = run_one_dataset(name, cfg, model, device, args)
            all_curves[name] = acc
            all_diag[name] = diag

            ds_dir = out_root / "per_dataset" / name
            ds_dir.mkdir(parents=True, exist_ok=True)
            np.save(ds_dir / "pred_delta_by_iter.npy", art["pred_delta_by_iter"])
            if art.get("preds_full_by_iter") is not None:
                outp = ds_dir / "preds_full_by_iter.npy"
                np.save(outp, art["preds_full_by_iter"])
                print(f"  [save-cell-preds-by-iter] {outp}")
            pd.DataFrame(
                {
                    "gene": art["genes"],
                    "true_delta": art["true_delta"],
                    "pred_delta": art["pred_delta_by_iter"][-1],
                    "true_sign": direction_signs(art["true_delta"], args.acc_eps),
                    "pred_sign": direction_signs(art["pred_delta_by_iter"][-1], args.acc_eps),
                }
            ).assign(
                correct_sign=lambda d: (d["true_sign"] != 0) & (d["true_sign"] == d["pred_sign"]),
                is_mapped=True,
                eligible_direction=lambda d: d["true_sign"] != 0,
                in_top_eval=[i in set(art["top_idx"].tolist()) for i in range(len(art["genes"]))],
            ).to_csv(ds_dir / "per_gene_final_changes.csv", index=False)

            save_trajectory_plot(
                ds_dir,
                name,
                art["pred_delta_by_iter"],
                art["true_delta"],
                args.save_trajectory_plot,
                args.trajectory_embed,
                args.seed,
            )

            print(f"  Final balanced accuracy: {acc[-1]:.2%} | inverted-truth BA: {diag.get('final_acc_inv_truth', 0):.2%}")
        except Exception as e:
            errors[name] = str(e)
            print(f"  [ERROR] {e}")

    save_accuracy_curves(out_root, all_curves, float(args.acc_eps))
    with open(out_root / "diagnostics.json", "w") as f:
        json.dump(all_diag, f, indent=2)
    with open(out_root / "errors.json", "w") as f:
        json.dump(errors, f, indent=2)

    print("\n" + "=" * 70)
    print(f"SUMMARY | scPRINT | outdir={out_root}")
    for name, acc in all_curves.items():
        print(f"{name:<12} | Balanced accuracy: {acc[-1]:.2%}")
    if errors:
        print("FAILED:", errors)


if __name__ == "__main__":
    main()
