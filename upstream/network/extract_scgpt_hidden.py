
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import json
import os
from pathlib import Path
import sys
import warnings
import datetime

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import scgpt as scg
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.preprocess import binning as scgpt_binning
from scgpt.utils import set_seed

# ====================  ====================
os.environ["KMP_WARNINGS"] = "off"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
warnings.filterwarnings("ignore")

# ==================== CLI / () ====================
import argparse
from typing import Optional


def _env_or(argval: Optional[str], env_key: str) -> Optional[str]:
    return argval if argval not in (None, "") else os.environ.get(env_key)


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract scGPT hidden-state gene embeddings and export cosine-edge TSVs (Gene1/Gene2/EdgeWeight)."
    )
    p.add_argument(
        "--input-root",
        type=str,
        default=os.environ.get("FOUNDBENCH_INPUT_ROOT"),
        help="Input root directory containing CHIP/Non_CHIP/STRING subfolders with *ExpressionData.csv files.",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default=os.environ.get("FOUNDBENCH_OUTPUT_ROOT"),
        help="Output root directory. Each dataset writes one {MODEL_NAME}_{DATASET}.tsv plus run_params.json.",
    )
    p.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("FOUNDBENCH_SCGPT_MODEL_DIR"),
        help="scGPT model directory containing vocab.json, args.json, best_model.pt.",
    )
    p.add_argument(
        "--folders",
        type=str,
        nargs="+",
        default=["CHIP"],
        help='Subfolders under input-root to process (e.g. CHIP Non_CHIP STRING). Default: ["CHIP"].',
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--batch-size", type=int, default=8, help="Forward batch size.")
    p.add_argument("--seq-topk-genes", type=int, default=512, help="Top-K expressed genes per cell to build sequences.")
    p.add_argument("--max-seq-len", type=int, default=512, help="Maximum sequence length.")
    p.add_argument("--n-cells", type=int, default=None, help="Use only N sampled cells (default: all).")
    p.add_argument("--use-log1p", action="store_true", help="Apply log1p to expression values before ranking.")
    p.add_argument("--no-save-all-edges", action="store_true", help="Disable all-edges export (use topK per gene).")
    p.add_argument("--topk-per-gene", type=int, default=1000, help="When not saving all edges, topK edges per Gene1.")
    p.add_argument("--clear-cache-every", type=int, default=10, help="Clear CUDA cache every N batches.")
    p.add_argument(
        "--evaluation-layout", action="store_true",
        help="Write output-root/scgpt_hidden/scGPT_DATASET.tsv for evaluate_aupr.py/evaluate_epr.py; requires one --folders value.",
    )
    args = p.parse_args()

    if not args.input_root or not args.output_root or not args.model_dir:
        p.error(
            "Missing required paths. Provide --input-root/--output-root/--model-dir "
            "or set env FOUNDBENCH_INPUT_ROOT/FOUNDBENCH_OUTPUT_ROOT/FOUNDBENCH_SCGPT_MODEL_DIR."
        )
    if args.batch_size < 1 or args.seq_topk_genes < 1 or args.max_seq_len < 1:
        p.error("--batch-size, --seq-topk-genes, and --max-seq-len must be positive")
    if args.n_cells is not None and args.n_cells < 1:
        p.error("--n-cells must be positive")
    if args.clear_cache_every < 1 or args.topk_per_gene < 1:
        p.error("--clear-cache-every and --topk-per-gene must be positive")
    if args.evaluation_layout and len(args.folders) != 1:
        p.error("--evaluation-layout requires exactly one --folders value to avoid overwriting datasets")
    return args


MODEL_NAME = "scGPT"

PAD_TOKEN = "<pad>"
N_BINS = 51
PAD_VALUE = -2

CONFIG = {}

# ==================== 1.  scGPT Model() ====================
def init_scgpt_model(model_dir: Path, seed: int):
    set_seed(seed)

    vocab_file = model_dir / "vocab.json"
    if not vocab_file.exists():
        raise FileNotFoundError(f"Vocabularydoes not exist: {vocab_file}")
    vocab = GeneVocab.from_file(vocab_file)

    model_config_file = model_dir / "args.json"
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)

    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_bins = int(model_configs.get("n_bins", N_BINS))
    if model_configs.get("input_style") != "binned" or n_bins < 2:
        raise ValueError("This extractor requires a checkpoint trained with binned expression input")
    append_cls = bool(model_configs.get("USE_CLS", not model_configs.get("no_cls", False)))
    required_tokens = [PAD_TOKEN] + (["<cls>"] if append_cls else [])
    missing_tokens = [token for token in required_tokens if token not in vocab]
    if missing_tokens:
        raise ValueError(f"Checkpoint vocabulary lacks required tokens: {missing_tokens}")
    model_configs["extract_append_cls"] = append_cls

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    ntokens = len(vocab)
    model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        vocab=vocab,
        pad_value=int(model_configs.get("pad_value", PAD_VALUE)),
        input_emb_style=model_configs.get("input_emb_style", "continuous"),
        n_input_bins=n_bins,
        dropout=float(model_configs.get("dropout", 0.1)),
        do_mvc=bool(model_configs.get("MVC", False)),
        use_fast_transformer=bool(model_configs.get("fast_transformer", False)) and device.type == "cuda",
    )

    model_file = model_dir / "best_model.pt"
    state_dict = torch.load(model_file, map_location="cpu")
    if isinstance(state_dict, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint is not a state dictionary: {model_file}")
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    model_state = model.state_dict()
    # scGPT's flash-attn backend stores packed QKV as Wqkv.{weight,bias};
    # PyTorch MultiheadAttention stores the same packed tensors as in_proj_{weight,bias}.
    for target_key in model_state:
        if target_key in state_dict:
            continue
        if ".self_attn.in_proj_weight" in target_key:
            source_key = target_key.replace(".self_attn.in_proj_weight", ".self_attn.Wqkv.weight")
        elif ".self_attn.in_proj_bias" in target_key:
            source_key = target_key.replace(".self_attn.in_proj_bias", ".self_attn.Wqkv.bias")
        elif ".self_attn.Wqkv.weight" in target_key:
            source_key = target_key.replace(".self_attn.Wqkv.weight", ".self_attn.in_proj_weight")
        elif ".self_attn.Wqkv.bias" in target_key:
            source_key = target_key.replace(".self_attn.Wqkv.bias", ".self_attn.in_proj_bias")
        else:
            continue
        if source_key in state_dict:
            state_dict[target_key] = state_dict[source_key]
    matched = {
        key: value for key, value in state_dict.items()
        if key in model_state and hasattr(value, "shape") and value.shape == model_state[key].shape
    }
    backbone_prefixes = ("encoder.", "value_encoder.", "transformer_encoder.", "bn.")
    missing_backbone = [
        key for key in model_state
        if key.startswith(backbone_prefixes) and key not in matched
    ]
    if missing_backbone:
        raise RuntimeError(
            f"Checkpoint does not fully cover the scGPT encoder ({len(missing_backbone)} missing/mismatched keys): "
            f"{missing_backbone[:8]}"
        )
    model.load_state_dict(matched, strict=False)
    print(f"[INFO] Loaded pretrained parameters: {len(matched)}/{len(model_state)}; encoder complete")

    model = model.to(device)
    model.eval()

    return model, vocab, device, model_configs


MODEL = None
VOCAB = None
DEVICE = None
MODEL_CONFIGS = None

# ==================== 2.  ====================
def extract_dataset_name(file_path: str) -> str:
    base = os.path.basename(file_path)
    for suffix in ["_chip_matched-ExpressionData.csv", "_processed-ExpressionData.csv", "-ExpressionData.csv"]:
        if suffix in base:
            return base.split(suffix)[0]
    return base.split(".csv")[0]


def clear_gpu_cache(force=False):
    if DEVICE is not None and DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        if force:
            torch.cuda.ipc_collect()
        #  log,
        # print("🔧 GPU")


def read_expression_matrix(path: str) -> pd.DataFrame:
    """
    : df [cells x genes]
    :=gene, =cell(index_col=0) -> 
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expression file not found: {path}")

    df = None
    try:
        df = pd.read_csv(path, sep="\t", header=0, index_col=0)
        if df.shape[1] == 0:
            df = None
    except Exception:
        df = None

    if df is None:
        df = pd.read_csv(path, sep=None, engine="python", header=0, index_col=0)

    df.index = df.index.astype(str).str.strip()
    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep="first")]

    # to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.fillna(0.0)

    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError(f"Empty expression matrix after parsing: {df.shape}")

    # :cells x genes
    df = df.T
    print(f"[INFO] Expression loaded: {df.shape[0]} cells x {df.shape[1]} genes")
    return df


# ==================== 3.  ====================
def build_gene_sequences(expr_df: pd.DataFrame, vocab: GeneVocab, config: dict):
    """
    expr_df: [cells x genes]
    Returns:
      sequences: List[List[int]]  gene token ids (no CLS yet)
      value_seqs: List[List[float]] expression values aligned to sequences
      valid_gene_symbols: List[str]  original symbols (for output)
      token_to_gene: Dict[int, str]
      used_cells: int
    """
    genes = expr_df.columns.tolist()
    expr_matrix = expr_df.values.astype(np.float32)
    if not np.isfinite(expr_matrix).all() or (expr_matrix < 0).any():
        raise ValueError("Expression matrix must contain finite, non-negative values for scGPT binning")
    n_cells, n_genes = expr_matrix.shape
    print(f"[INFO] Expression matrix: {n_cells} cells x {n_genes} genes")

    # gene -> token / token -> gene (case-insensitive match to scGPT vocab)
    gene_to_token = {}
    token_to_gene = {}
    valid_gene_idx = []

    for i, gene in enumerate(genes):
        g_key = str(gene).strip().upper()
        if g_key in vocab:
            tid = int(vocab[g_key])
            if tid in token_to_gene:
                raise ValueError(f"Multiple expression columns map to scGPT token {tid}: {token_to_gene[tid]}, {gene}")
            gene_to_token[gene] = tid
            token_to_gene[tid] = gene
            valid_gene_idx.append(i)

    print(f"[INFO] Matched genes: {len(valid_gene_idx)}/{n_genes}")
    if len(valid_gene_idx) == 0:
        raise ValueError("No matched genes. Cannot continue.")

    expr_matrix = expr_matrix[:, valid_gene_idx]
    valid_gene_symbols = [genes[i] for i in valid_gene_idx]

    if config["USE_LOG1P"]:
        expr_matrix = np.log1p(np.maximum(expr_matrix, 0.0))

    # limit cells
    if config["N_CELLS"] is not None and int(config["N_CELLS"]) < n_cells:
        idx = np.random.choice(n_cells, int(config["N_CELLS"]), replace=False)
        expr_matrix = expr_matrix[idx, :]
        n_cells_use = int(config["N_CELLS"])
    else:
        n_cells_use = n_cells

    topk = min(int(config["SEQ_TOPK_GENES"]), int(config["MAX_SEQ_LEN"]), len(valid_gene_symbols))

    sequences = []
    value_seqs = []
    for cell_idx in range(n_cells_use):
        row = expr_matrix[cell_idx]
        expressed = np.flatnonzero(row > 0)
        if expressed.size == 0:
            continue
        # scGPT's own binning assigns zero to absent genes and quantile bins 1..n_bins-1
        # to positive genes. Bin the full matched row before selecting sequence tokens.
        binned = scgpt_binning(row, n_bins=int(config["N_BINS"]))
        idxs = expressed[np.argsort(-row[expressed], kind="stable")[:topk]]
        sequences.append([gene_to_token[valid_gene_symbols[i]] for i in idxs])
        value_seqs.append([float(binned[i]) for i in idxs])

    if not sequences:
        raise ValueError("No cells contain positive expression for vocabulary-matched genes")
    if len(sequences) < n_cells_use:
        print(f"[WARN] Skipped {n_cells_use - len(sequences)} all-zero cells")
    return sequences, value_seqs, valid_gene_symbols, token_to_gene, len(sequences)


# ==================== 4. Hidden () ====================
def extract_hidden_embeddings(model, sequences, value_seqs, gene_symbols, token_to_gene, vocab, config):
    """
    :
      gene_embeddings: [n_final_genes, hidden_dim]
      final_gene_list
    """
    hidden_dim = int(MODEL_CONFIGS["embsize"])
    batch_size = int(config["BATCH_SIZE"])
    n_batches = (len(sequences) + batch_size - 1) // batch_size

    # :token_id -> sum/cnt
    sum_vec = {}
    cnt = {}

    pad_id = int(vocab[PAD_TOKEN])
    append_cls = bool(config["APPEND_CLS"])
    cls_id = int(vocab["<cls>"]) if append_cls else None
    pad_value = float(config["PAD_VALUE"])

    print(f"\n[INFO] Forward scGPT (batch_size={batch_size}, total_batches={n_batches})")
    print(f"[INFO] Using checkpoint-matched binned values; append_cls={append_cls}")

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
        for batch_idx in tqdm(range(n_batches), desc="Forward scGPT"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(sequences))
            batch_seqs = sequences[start_idx:end_idx]
            batch_vals = value_seqs[start_idx:end_idx]
            if not batch_seqs:
                continue

            max_len = max(len(seq) for seq in batch_seqs) + int(append_cls)
            input_ids = []
            values = []

            for seq, vals in zip(batch_seqs, batch_vals):
                ids = ([cls_id] if append_cls else []) + list(seq)
                vs = ([0.0] if append_cls else []) + list(vals)
                pad_len = max_len - len(ids)
                input_ids.append(ids + [pad_id] * pad_len)
                values.append(vs + [pad_value] * pad_len)

            input_ids = torch.tensor(input_ids, dtype=torch.long, device=DEVICE)
            values = torch.tensor(values, dtype=torch.float, device=DEVICE)

            src_key_padding_mask = input_ids.eq(pad_id)
            transformer_output = model._encode(input_ids, values, src_key_padding_mask)
            hidden_states = transformer_output.float().detach().cpu().numpy()  # [B, L, H]
            ids_np = input_ids.detach().cpu().numpy()
            mask_np = (~src_key_padding_mask).detach().cpu().numpy()  # True for valid

            B, L, H = hidden_states.shape
            for bi in range(B):
                for li in range(L):
                    if not mask_np[bi, li]:
                        break
                    tid = int(ids_np[bi, li])
                    if tid == pad_id or tid == cls_id:
                        continue
                    vec = hidden_states[bi, li].astype(np.float64)
                    if tid not in sum_vec:
                        sum_vec[tid] = vec
                        cnt[tid] = 1
                    else:
                        sum_vec[tid] += vec
                        cnt[tid] += 1

            if (batch_idx + 1) % int(config["CLEAR_CACHE_EVERY"]) == 0:
                clear_gpu_cache()

    # token -> gene -> embedding
    gene_emb = {}
    for tid, vec in sum_vec.items():
        gene = token_to_gene.get(int(tid), None)
        if gene is None:
            continue
        gene_emb[gene] = (vec / max(cnt.get(tid, 1), 1)).astype(np.float32)

    final_gene_list = sorted(gene_emb.keys())
    gene_embeddings = np.stack([gene_emb[g] for g in final_gene_list], axis=0)

    print(f"[INFO] Extraction complete | {len(final_gene_list)} genes x {hidden_dim} dim")

    clear_gpu_cache(force=True)
    return gene_embeddings, final_gene_list


# ==================== 5. :() ====================
def compute_and_save_all_cosine_edges_stream(
    gene_list, embeddings, output_tsv, save_all_edges=True, topk_per_gene=1000
):
    """
     i!=j (),:Gene1 Gene2 EdgeWeight
    """
    gene_list = list(gene_list)
    emb = np.asarray(embeddings, dtype=np.float32)
    n_genes = len(gene_list)
    if n_genes < 2:
        raise ValueError("Need at least 2 genes to compute edges.")
    if not np.isfinite(emb).all():
        raise ValueError("Hidden embeddings contain non-finite values")

    denom = np.linalg.norm(emb, axis=1, keepdims=True)
    if (denom == 0).any():
        raise ValueError("Hidden embeddings contain zero-norm vectors")
    E = emb / denom

    output_tsv = Path(output_tsv)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)

    total_edges = 0
    with open(output_tsv, "w", encoding="utf-8") as f:
        f.write("Gene1\tGene2\tEdgeWeight\n")
        for i in tqdm(range(n_genes), desc="All cosine edges (stream)"):
            sims = E[i] @ E.T
            sims[i] = -np.inf
            g1 = gene_list[i]
            if save_all_edges:
                targets = (j for j in range(n_genes) if j != i)
            else:
                k = min(int(topk_per_gene), n_genes - 1)
                selected = np.argpartition(-sims, k - 1)[:k]
                targets = sorted(selected, key=lambda j: (-float(sims[j]), gene_list[j]))
            for j in targets:
                f.write(f"{g1}\t{gene_list[j]}\t{float(sims[j]):.8f}\n")
                total_edges += 1

    return {
        "mode": "all_stream" if save_all_edges else "topk_per_gene",
        "n_genes": int(n_genes),
        "n_edges": int(total_edges),
        "topk_per_gene": None if save_all_edges else int(topk_per_gene),
    }


# ==================== 6. Dataset( IO ) ====================
def process_single_dataset(expr_path: str, folder_name: str, config: dict):
    start_time = datetime.datetime.now()
    dataset_name = extract_dataset_name(expr_path)

    # Evaluation layout is compatible with evaluate_aupr.py/evaluate_epr.py.
    out_dir = (OUTPUT_ROOT / "scgpt_hidden") if config["EVALUATION_LAYOUT"] else (OUTPUT_ROOT / folder_name / dataset_name)
    out_dir.mkdir(exist_ok=True, parents=True)

    record = {
        "Run_Datetime": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "Model_Name": MODEL_NAME,
        "Folder_Name": folder_name,
        "Dataset_Name": dataset_name,
        "Input_File": str(expr_path),
        "Input_Genes_Count": 0,
        "Matched_Genes_Count": 0,
        "Final_Genes_Count": 0,
        "Embedding_Dim": 0,
        "Total_Edges_Generated": 0,
        "Output_TSV_Path": "",
        "Process_Status": "Success",
        "Process_Time_Seconds": 0.0,
        "Error_Message": "",
    }

    try:
        print(f"\n{'='*60}")
        print(f"[INFO] Start dataset: {dataset_name} | Folder: {folder_name}")
        print(f"[INFO] Input file: {expr_path}")

        # 1) read expression -> cells x genes
        df = read_expression_matrix(expr_path)
        record["Input_Genes_Count"] = int(df.shape[1])

        # 2) build sequences (token ids + scGPT-binned expression values)
        sequences, value_seqs, gene_symbols, token_to_gene, n_cells_used = build_gene_sequences(df, VOCAB, config)
        record["Matched_Genes_Count"] = int(len(gene_symbols))

        # 3) hidden embeddings (in memory)
        gene_embeddings, final_gene_list = extract_hidden_embeddings(
            MODEL, sequences, value_seqs, gene_symbols, token_to_gene, VOCAB, config
        )
        record["Final_Genes_Count"] = int(len(final_gene_list))
        record["Embedding_Dim"] = int(gene_embeddings.shape[1])

        # 4) only edges TSV: {MODEL_NAME}_{DATASET}.tsv
        edge_tsv = out_dir / f"{MODEL_NAME}_{dataset_name}.tsv"
        cosine_info = compute_and_save_all_cosine_edges_stream(
            final_gene_list, gene_embeddings, edge_tsv,
            save_all_edges=config["SAVE_ALL_EDGES"], topk_per_gene=config["TOPK_PER_GENE"],
        )
        record["Total_Edges_Generated"] = int(cosine_info["n_edges"])
        record["Output_TSV_Path"] = str(edge_tsv)
        print(f"[INFO] Edge file saved: {edge_tsv}")

        # 5) save run_params.json
        run_params = {
            "dataset": dataset_name,
            "folder": folder_name,
            "input_file": str(expr_path),
            "output_dir": str(out_dir),
            "outputs": {"edges_tsv": str(edge_tsv)},
            "model": {
                "model_name": MODEL_NAME,
                "model_dir": str(config["MODEL_DIR"]),
                "embsize": int(MODEL_CONFIGS.get("embsize", gene_embeddings.shape[1])),
                "nheads": int(MODEL_CONFIGS.get("nheads", -1)),
                "nlayers": int(MODEL_CONFIGS.get("nlayers", -1)),
                "d_hid": int(MODEL_CONFIGS.get("d_hid", -1)),
                "pad_token": PAD_TOKEN,
                "pad_value": int(config["PAD_VALUE"]),
                "n_bins": int(config["N_BINS"]),
            },
            "sequence": {
                "seq_topk_genes": int(config["SEQ_TOPK_GENES"]),
                "max_seq_len": int(config["MAX_SEQ_LEN"]),
                "use_log1p": bool(config["USE_LOG1P"]),
                "n_cells": config["N_CELLS"],
                "n_cells_used": int(n_cells_used),
                "batch_size": int(config["BATCH_SIZE"]),
                "append_cls": bool(config["APPEND_CLS"]),
                "value_preprocessing": "scgpt.preprocess.binning on full matched row before top-gene selection",
                "gene_match": "case_insensitive_upper",
                "evaluation_layout": bool(config["EVALUATION_LAYOUT"]),
            },
            "stats": {
                "n_input_genes": int(record["Input_Genes_Count"]),
                "n_matched_genes": int(record["Matched_Genes_Count"]),
                "n_final_genes_with_embedding": int(record["Final_Genes_Count"]),
                "embedding_dim": int(record["Embedding_Dim"]),
            },
            "cosine_export": cosine_info,
            "processing_time_seconds": round((datetime.datetime.now() - start_time).total_seconds(), 2),
        }

        params_path = out_dir / (f"{dataset_name}_run_params.json" if config["EVALUATION_LAYOUT"] else "run_params.json")
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(run_params, f, indent=2, ensure_ascii=False)

        record["Process_Time_Seconds"] = run_params["processing_time_seconds"]
        print(f"[INFO] Dataset done | elapsed: {record['Process_Time_Seconds']}s")

    except Exception as e:
        traceback_msg = str(e)[:300]
        record["Process_Status"] = "Failed"
        record["Error_Message"] = traceback_msg
        print(f"[ERROR] Failed: {dataset_name} | {traceback_msg}")
        import traceback
        traceback.print_exc()

    return record


# ==================== 7. () ====================
def main():
    global MODEL, VOCAB, DEVICE, MODEL_CONFIGS, CONFIG
    args = parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    model_dir = Path(args.model_dir)
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input root does not exist: {input_root}")
    if not model_dir.is_dir():
        raise NotADirectoryError(f"scGPT model directory does not exist: {model_dir}")

    CONFIG = {
        "SEQ_TOPK_GENES": int(args.seq_topk_genes),
        "MAX_SEQ_LEN": int(args.max_seq_len),
        "N_CELLS": args.n_cells,
        "USE_LOG1P": bool(args.use_log1p),
        "BATCH_SIZE": int(args.batch_size),
        "SAVE_ALL_EDGES": (not bool(args.no_save_all_edges)),
        "TOPK_PER_GENE": int(args.topk_per_gene),
        "CLEAR_CACHE_EVERY": int(args.clear_cache_every),
        "MODEL_DIR": str(model_dir),
        "EVALUATION_LAYOUT": bool(args.evaluation_layout),
    }

    output_root.mkdir(exist_ok=True, parents=True)
    print(f"\n[INFO] Start {MODEL_NAME} hidden states batch processing")
    print(f"[INFO] Input root: {input_root}")
    print(f"[INFO] Output root: {output_root}")
    print(f"📦 Model: {model_dir}")
    print(f"[INFO] Folders: {args.folders}")
    print(f"[INFO] Current config: {json.dumps({k:v for k,v in CONFIG.items() if k!='MODEL_DIR'}, indent=2, ensure_ascii=False)}")

    # init model once
    MODEL, VOCAB, DEVICE, MODEL_CONFIGS = init_scgpt_model(model_dir=model_dir, seed=int(args.seed))
    CONFIG["N_BINS"] = int(MODEL_CONFIGS["n_bins"])
    CONFIG["PAD_VALUE"] = int(MODEL_CONFIGS.get("pad_value", PAD_VALUE))
    CONFIG["APPEND_CLS"] = bool(MODEL_CONFIGS["extract_append_cls"])

    # override module-level OUTPUT_ROOT/INPUT_ROOT behavior with local paths
    # (keep downstream functions unchanged by binding to globals here)
    globals()["INPUT_ROOT"] = input_root
    globals()["OUTPUT_ROOT"] = output_root

    all_records = []
    for folder in args.folders:
        folder_path = input_root / folder
        if not folder_path.exists():
            print(f"\n[WARN] Folder not found, skip: {folder_path}")
            continue

        print(f"\n{'='*60}")
        print(f"[INFO] Processing folder: {folder}")

        if folder == "CHIP":
            expr_files = list(folder_path.glob("*_chip_matched-ExpressionData.csv"))
        else:
            expr_files = list(folder_path.glob("*_processed-ExpressionData.csv"))

        print(f"[INFO] Found {len(expr_files)} expression files")

        for expr_file in expr_files:
            rec = process_single_dataset(str(expr_file), folder, CONFIG)
            all_records.append(rec)
            clear_gpu_cache(force=True)

    if all_records:
        summary_df = pd.DataFrame(all_records)
        summary_path = output_root / f"{MODEL_NAME}_processing_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n[INFO] Summary saved: {summary_path}")

        success_count = sum(1 for r in all_records if r["Process_Status"] == "Success")
        fail_count = len(all_records) - success_count
        print(f"\n[INFO] Stats: success {success_count} / fail {fail_count}")
        if fail_count:
            raise SystemExit(1)
    else:
        raise RuntimeError("No expression files were found in the requested folders")

    print(f"\n{'='*60}")
    print(f"[INFO] Batch finished. Outputs saved in: {output_root}")


if __name__ == "__main__":
    main()
