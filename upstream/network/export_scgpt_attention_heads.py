import argparse
import json
import os
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from anndata import AnnData
from einops import rearrange
from scipy.sparse import issparse
from tqdm import tqdm

from utils_heads import parse_head_indices


def parse_args():
    p = argparse.ArgumentParser(description="Export scGPT per-head TSV files.")
    p.add_argument("--data-type", required=True, choices=["CHIP", "Non_CHIP", "STRING"])
    p.add_argument("--dataset", required=True, type=str)
    p.add_argument("--input-root", required=True, type=str)
    p.add_argument("--output-root", required=True, type=str)
    p.add_argument("--scgpt-repo-dir", required=True, type=str)
    p.add_argument("--model-dir", required=True, type=str)
    p.add_argument("--batch-size", default=8, type=int)
    p.add_argument("--target-layer", default=11, type=int)
    p.add_argument(
        "--head-indices",
        default="all",
        type=str,
        help="Comma-separated head indices to export, e.g. '0,3,7'. Default 'all' exports every head.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")
    os.environ["KMP_WARNINGS"] = "off"

    sys.path.insert(0, args.scgpt_repo_dir)
    import scgpt as scg  # noqa: F401
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import tokenize_and_pad_batch
    from scgpt.preprocess import binning as scgpt_binning
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
    from scgpt.utils import set_seed
    from extract_scgpt_hidden import init_scgpt_model

    input_root = Path(args.input_root)
    output_root = Path(args.output_root) / "scgpt"
    output_root.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)

    suffix = "_chip_matched" if args.data_type == "CHIP" else "_processed"
    subdir = "CHIP" if args.data_type == "CHIP" else args.data_type
    expr_file = input_root / subdir / f"{args.dataset}{suffix}-ExpressionData.csv"
    if not expr_file.exists():
        raise FileNotFoundError(expr_file)

    model, vocab, device, mc = init_scgpt_model(model_dir, 42)
    append_cls = bool(mc["extract_append_cls"])
    if not 0 <= int(args.target_layer) < int(mc["nlayers"]):
        raise ValueError(f"target-layer must be in 0..{int(mc['nlayers']) - 1}")

    df = pd.read_csv(expr_file, index_col=0).T
    genes = df.columns.tolist()
    matched_idx = [j for j, gene in enumerate(genes) if str(gene).strip().upper() in vocab]
    if not matched_idx:
        raise ValueError("No expression genes match the scGPT vocabulary")
    df = df.iloc[:, matched_idx]
    genes = df.columns.tolist()
    gene_keys = [str(gene).strip().upper() for gene in genes]
    if len(gene_keys) != len(set(gene_keys)):
        raise ValueError("Expression genes map to duplicate scGPT tokens")
    adata = AnnData(X=df.values, obs=pd.DataFrame(index=df.index), var=pd.DataFrame(index=genes))
    all_counts = adata.X.A if issparse(adata.X) else adata.X
    all_counts = np.asarray(all_counts, dtype=np.float32)
    if not np.isfinite(all_counts).all() or (all_counts < 0).any():
        raise ValueError("scGPT binning requires finite, non-negative expression")
    all_counts = np.stack([scgpt_binning(row, n_bins=int(mc["n_bins"])) for row in all_counts])
    gene_ids = np.array([vocab[g] for g in gene_keys], dtype=int)
    tok = tokenize_and_pad_batch(
        all_counts,
        gene_ids,
        max_len=len(genes) + int(append_cls),
        vocab=vocab,
        pad_token="<pad>",
        pad_value=int(mc.get("pad_value", -2)),
        append_cls=append_cls,
        include_zero_gene=True,
    )
    all_gene_ids, all_values = tok["genes"], tok["values"]
    pad_mask = all_gene_ids.eq(vocab["<pad>"])
    n_heads = int(mc["nheads"])

    attn_sum = np.zeros((n_heads, len(genes), len(genes)), dtype=np.float64)
    num_cells = 0

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        for i in tqdm(range(0, all_gene_ids.size(0), int(args.batch_size)), desc="Extracting scGPT heads"):
            bsz = min(int(args.batch_size), all_gene_ids.size(0) - i)
            bg = all_gene_ids[i : i + bsz].to(device)
            bv = all_values[i : i + bsz].to(device)
            bm = pad_mask[i : i + bsz].to(device)
            src = model.encoder(bg)
            val = model.value_encoder(bv)
            x = src + val
            if hasattr(model, "bn"):
                x = model.bn(x.permute(0, 2, 1)).permute(0, 2, 1)

            for lidx in range(int(args.target_layer)):
                layer = model.transformer_encoder.layers[lidx]
                qkv = layer.self_attn.Wqkv(x) if hasattr(layer.self_attn, "Wqkv") else torch.nn.functional.linear(x, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias)
                qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=n_heads)
                q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
                q = q.permute(0, 2, 1, 3)
                k = k.permute(0, 2, 1, 3)
                v = v.permute(0, 2, 1, 3)
                scale = 1.0 / np.sqrt(q.shape[-1])
                w = torch.matmul(q, k.transpose(-2, -1)) * scale
                w = w.masked_fill(bm.unsqueeze(1).unsqueeze(2), -65504.0)
                p = torch.softmax(w, dim=-1)
                o = torch.matmul(p, v).permute(0, 2, 1, 3).reshape(bsz, x.size(1), -1)
                o = layer.self_attn.out_proj(o)
                x = layer.norm1(x + o)
                ff = layer.linear2(layer.activation(layer.linear1(x)))
                x = layer.norm2(x + ff)

            fl = model.transformer_encoder.layers[int(args.target_layer)]
            qkv = fl.self_attn.Wqkv(x) if hasattr(fl.self_attn, "Wqkv") else torch.nn.functional.linear(x, fl.self_attn.in_proj_weight, fl.self_attn.in_proj_bias)
            qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=n_heads)
            q, k = qkv[:, :, 0], qkv[:, :, 1]
            s = q.permute(0, 2, 1, 3) @ k.permute(0, 2, 3, 1)
            s = s.masked_fill(bm.unsqueeze(1).unsqueeze(2), -65504.0)
            p = torch.softmax(s * (1.0 / np.sqrt(q.shape[-1])), dim=-1).detach().cpu().numpy()  # [b,h,m,m]
            if append_cls:
                p = p[:, :, 1:, 1:]  # drop cls only when used
            attn_sum += p.sum(axis=0)
            num_cells += bsz
            if device.type == "cuda":
                torch.cuda.empty_cache()

    attn_avg = attn_sum / float(num_cells)
    head_list = parse_head_indices(args.head_indices, n_heads)
    for h in head_list:
        mat = attn_avg[h]
        edges = pd.DataFrame(mat, index=genes, columns=genes).stack().reset_index()
        edges.columns = ["Gene1", "Gene2", "EdgeWeight"]
        edges = edges[edges["Gene1"] != edges["Gene2"]]
        edges = edges.sort_values("EdgeWeight", ascending=False).reset_index(drop=True)
        out = output_root / f"scgpt_{args.dataset}_head{h}.tsv"
        edges.to_csv(out, sep="\t", index=False)
        print(f"[INFO] Saved {out} edges={len(edges)}")


if __name__ == "__main__":
    main()
