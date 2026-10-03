import copy
import json
import os
from pathlib import Path
import sys
import warnings
import datetime
import glob

import torch
from anndata import AnnData
import scanpy as sc
import numpy as np
import pandas as pd
from tqdm import tqdm
from einops import rearrange

sys.path.insert(0, os.environ.get("FOUNDBENCH_SCGPT_REPO_DIR", "/mnt/10T/yzn/benchmark_GRN/pre_scgpt/scGPT"))

import scgpt as scg
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.utils import set_seed 
from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.preprocess import binning as scgpt_binning
from scipy.sparse import issparse
from extract_scgpt_hidden import init_scgpt_model as load_scgpt_encoder

os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

# ==================== Path defaults (Fig. 2 main setting = TFs+500) ====================
# Expression root: input_process (TFs+500). Use input_process1000 only for the TFs+1000 ablation.
INPUT_ROOT = Path(os.environ.get("FOUNDBENCH_INPUT_ROOT", "/mnt/10T/yzn/benchmark_GRN/input_process"))
# Attention outputs: output_att500 (evaluated copies live under evl_omipath/output_att500).
OUTPUT_ROOT = Path(os.environ.get("FOUNDBENCH_SCGPT_ATT_OUTPUT_ROOT", "/mnt/10T/yzn/benchmark_GRN/model/output_att500/scgpt"))
# Model()
MODEL_DIR = Path(os.environ.get("FOUNDBENCH_SCGPT_MODEL_DIR", "/mnt/10T/yzn/benchmark_GRN/pre_scgpt/scGPT/scgpt_human"))
# Model(scgpt_att)
MODEL_NAME = "scgpt_att"
# ()
SEED = 42
BATCH_SIZE = 8
ATTN_LAYERS = 11
PAD_TOKEN = "<pad>"
SPECIAL_TOKENS = [PAD_TOKEN, "<cls>", "<eoc>"]
N_BINS = 51
PAD_VALUE = -2

# ==================== 1. Model(,) ====================
def init_scgpt_model():
    """Load the checkpoint with the same validated encoder settings as hidden extraction."""
    return load_scgpt_encoder(MODEL_DIR, SEED)


def packed_qkv(attn, x):
    if hasattr(attn, "Wqkv"):
        return attn.Wqkv(x)
    return torch.nn.functional.linear(x, attn.in_proj_weight, attn.in_proj_bias)

# ==================== 2.  ====================
def get_all_expression_files():
    """INPUT_ROOT,ExpressionDataNetwork"""
    task_list = []
    # CHIP/Non_CHIP/STRING
    for data_type in ["CHIP", "Non_CHIP", "STRING"]:
        data_type_dir = INPUT_ROOT / data_type
        if not data_type_dir.exists():
            print(f"[WARN] Skip missing directory: {data_type_dir}")
            continue
        
        # ExpressionData
        if data_type == "CHIP":
            # CHIP:*_chip_matched-ExpressionData.csv
            expr_files = glob.glob(str(data_type_dir / "*_chip_matched-ExpressionData.csv"))
        else:
            # Non_CHIP/STRING:*_processed-ExpressionData.csv
            expr_files = glob.glob(str(data_type_dir / "*_processed-ExpressionData.csv"))
        
        for expr_file in expr_files:
            expr_file = Path(expr_file)
            # Dataset
            if data_type == "CHIP":
                dataset_name = expr_file.stem.replace("_chip_matched-ExpressionData", "")
            else:
                dataset_name = expr_file.stem.replace("_processed-ExpressionData", "")
            
            task_list.append({
                "data_type": data_type,
                "dataset_name": dataset_name,
                "expr_file": expr_file,
            })
    
    print(f"\n📋 :{len(task_list)}Dataset")
    for idx, task in enumerate(task_list):
        print(f"  {idx+1}. {task['data_type']}/{task['dataset_name']}")
    return task_list

# ==================== 3. Dataset ====================
def process_single_dataset(model, vocab, device, model_configs, task):
    """Dataset"""
    data_type = task["data_type"]
    dataset_name = task["dataset_name"]
    expr_file = task["expr_file"]
    
    # (:OUTPUT_ROOT/data_type/)
    output_dir = OUTPUT_ROOT / data_type
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # (Model+Dataset:scgpt_att_Dataset)
    file_prefix = f"{MODEL_NAME}_{dataset_name}"
    
    try:
        # ====================  ====================
        print(f"\n[INFO] Start processing: {data_type}/{dataset_name}")
        print(f"   :{expr_file}")
        
        # 
        df = pd.read_csv(expr_file, index_col=0)
        df = df.T  # =,=
        genes = df.columns.tolist()
        matched_idx = [j for j, gene in enumerate(genes) if str(gene).strip().upper() in vocab]
        if not matched_idx:
            raise ValueError("No expression genes match the scGPT vocabulary")
        df = df.iloc[:, matched_idx]
        genes = df.columns.tolist()
        gene_keys = [str(gene).strip().upper() for gene in genes]
        if len(gene_keys) != len(set(gene_keys)):
            raise ValueError("Expression genes map to duplicate scGPT tokens")
        all_counts = df.values
        
        # AnnData
        adata = AnnData(
            X=all_counts,
            obs=pd.DataFrame(index=df.index),
            var=pd.DataFrame(index=genes)
        )
        print(f"   :{adata.n_obs} | {adata.n_vars}")
        
        # ==================== Tokenization ====================
        all_counts = adata.X.A if issparse(adata.X) else adata.X
        all_counts = np.asarray(all_counts, dtype=np.float32)
        if not np.isfinite(all_counts).all() or (all_counts < 0).any():
            raise ValueError("scGPT binning requires finite, non-negative expression")
        all_counts = np.stack(
            [scgpt_binning(row, n_bins=int(model_configs["n_bins"])) for row in all_counts]
        )
        gene_ids = np.array([vocab[gene] for gene in gene_keys], dtype=int)
        append_cls = bool(model_configs["extract_append_cls"])
        
        tokenized_all = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=len(genes) + int(append_cls),
            vocab=vocab,
            pad_token=PAD_TOKEN,
            pad_value=int(model_configs.get("pad_value", PAD_VALUE)),
            append_cls=append_cls,
            include_zero_gene=True,
        )
        all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
        src_key_padding_mask = all_gene_ids.eq(vocab[PAD_TOKEN])
        
        # ==================== Attention (occurrence-weighted mean) ====================
        # Paper Methods: head-average final-layer attention, then aggregate gene pairs by
        # occurrence count across cells: EdgeWeight = sum(attn) / count(co-occurrences).
        if device.type == "cuda":
            torch.cuda.empty_cache()
        nhead = model_configs["nheads"]
        n_genes = len(genes)
        attention_sum = np.zeros((n_genes, n_genes), dtype=np.float64)
        attention_count = np.zeros((n_genes, n_genes), dtype=np.int64)
        num_cells = 0
        M = all_gene_ids.size(1)
        N = all_gene_ids.size(0)
        
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            for i in tqdm(range(0, N, BATCH_SIZE), desc=f"   Batch"):
                current_batch_size = min(BATCH_SIZE, N - i)
                
                # batch
                batch_gene_ids = all_gene_ids[i:i+current_batch_size].to(device)
                batch_values = all_values[i:i+current_batch_size].to(device)
                batch_mask = src_key_padding_mask[i:i+current_batch_size].to(device)
                
                # embedding
                src_embs = model.encoder(batch_gene_ids)
                val_embs = model.value_encoder(batch_values)
                total_embs = src_embs + val_embs
                
                # BatchNorm()
                if hasattr(model, 'bn'):
                    total_embs = model.bn(total_embs.permute(0,2,1)).permute(0,2,1)
                
                # Transformer
                for layer_idx in range(ATTN_LAYERS):
                    layer = model.transformer_encoder.layers[layer_idx]
                    
                    # Self-Attention
                    qkv = packed_qkv(layer.self_attn, total_embs)
                    qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=nhead)
                    q, k, v = qkv[:, :, 0, :, :], qkv[:, :, 1, :, :], qkv[:, :, 2, :, :]
                    
                    q = q.permute(0,2,1,3)
                    k = k.permute(0,2,1,3)
                    v = v.permute(0,2,1,3)
                    
                    # 
                    scale = 1.0 / np.sqrt(q.shape[-1])
                    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
                    if batch_mask is not None:
                        mask_expanded = batch_mask.unsqueeze(1).unsqueeze(2)
                        attn_weights = attn_weights.masked_fill(mask_expanded, -65504.0)
                    
                    attn_probs = torch.softmax(attn_weights, dim=-1)
                    
                    # 
                    attn_output = torch.matmul(attn_probs, v)
                    attn_output = attn_output.permute(0,2,1,3)
                    attn_output = attn_output.reshape(current_batch_size, M, -1)
                    attn_output = layer.self_attn.out_proj(attn_output)
                    
                    # +LayerNorm
                    total_embs = total_embs + attn_output
                    total_embs = layer.norm1(total_embs)
                    
                    # Feed-Forward
                    ffn_output = layer.linear2(layer.activation(layer.linear1(total_embs)))
                    total_embs = total_embs + ffn_output
                    total_embs = layer.norm2(total_embs)
                
                # Final-layer attention (head-averaged), then occurrence-weighted gene pairs
                final_layer = model.transformer_encoder.layers[ATTN_LAYERS]
                qkv = packed_qkv(final_layer.self_attn, total_embs)
                qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=nhead)
                q, k = qkv[:, :, 0, :, :], qkv[:, :, 1, :, :]
                
                # 
                attn_scores = q.permute(0,2,1,3) @ k.permute(0,2,3,1)
                if batch_mask is not None:
                    mask_expanded = batch_mask.unsqueeze(1).unsqueeze(2)
                    attn_scores = attn_scores.masked_fill(mask_expanded, -65504.0)
                
                scale = 1.0 / np.sqrt(q.shape[-1])
                attn_probs = torch.softmax(attn_scores * scale, dim=-1)
                attn_scores_final = attn_probs.mean(1).detach().cpu().numpy()  # [B, M, M]
                batch_mask_np = batch_mask.detach().cpu().numpy()  # True = pad

                for b in range(current_batch_size):
                    # Valid non-pad tokens excluding <cls> at position 0.
                    valid = ~batch_mask_np[b]
                    if append_cls:
                        valid[0] = False
                    gene_pos = np.where(valid)[0]
                    if gene_pos.size < 2:
                        continue
                    # include_zero_gene keeps the input gene order; CLS adds one offset.
                    gene_idx = gene_pos - int(append_cls)
                    if np.any(gene_idx < 0) or np.any(gene_idx >= n_genes):
                        keep = (gene_idx >= 0) & (gene_idx < n_genes)
                        gene_pos = gene_pos[keep]
                        gene_idx = gene_idx[keep]
                        if gene_idx.size < 2:
                            continue
                    cell_attn = attn_scores_final[b][np.ix_(gene_pos, gene_pos)]
                    attention_sum[np.ix_(gene_idx, gene_idx)] += cell_attn
                    attention_count[np.ix_(gene_idx, gene_idx)] += 1

                num_cells += current_batch_size
                
                # 
                if i % (BATCH_SIZE * 10) == 0:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
        
        gene_attention = np.divide(
            attention_sum,
            attention_count,
            out=np.zeros_like(attention_sum, dtype=np.float64),
            where=attention_count > 0,
        )        
        # ==================== TSV ====================
        # (,)
        interactions_df = pd.DataFrame(
            gene_attention, index=genes, columns=genes
        ).stack().reset_index()
        interactions_df.columns = ["Gene1", "Gene2", "EdgeWeight"]
        interactions_df = interactions_df[interactions_df["Gene1"] != interactions_df["Gene2"]]
        interactions_df = interactions_df.sort_values("EdgeWeight", ascending=False).reset_index(drop=True)
        
        # TSV (no extra filtering)
        output_tsv = output_dir / f"{file_prefix}.tsv"
        interactions_df.to_csv(output_tsv, sep="\t", index=False)
        
        # ==================== () ====================
        params = {
            "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Model": MODEL_NAME,
            "": data_type,
            "Dataset": dataset_name,
            "": str(expr_file),
            "": str(output_tsv),
            "": len(genes),
            "": adata.n_obs,
            "": len(interactions_df),
            "": len(interactions_df),
            "": "100.00%",
            "Aggregation": "occurrence_weighted_mean",
            "Value_Preprocessing": "scgpt.preprocess.binning",
            "N_Bins": int(model_configs["n_bins"]),
            "Append_CLS": append_cls,
            "Pairs_With_Count": int((attention_count > 0).sum()),
        }
        
        # 
        params_csv = output_dir / f"{MODEL_NAME}_run_parameters.csv"
        if not params_csv.exists():
            pd.DataFrame([params]).to_csv(params_csv, index=False, encoding="utf-8")
        else:
            pd.DataFrame([params]).to_csv(params_csv, mode="a", header=False, index=False, encoding="utf-8")
        
        print("[INFO] Processing complete:")
        print(f"   :{output_tsv}")
        return True
    
    except Exception as e:
        print(f"[ERROR] Failed {data_type}/{dataset_name}: {str(e)[:200]}")
        return False

# ==================== 4. () ====================
def main():
    print("="*80)
    print("📦 scGPT(TSV)")
    print(f"Model:{MODEL_NAME}")
    print(f"Input root:{INPUT_ROOT}")
    print(f"Output root:{OUTPUT_ROOT}")
    print("="*80)
    
    # 1. Model()
    print("\n🔧 scGPTModel...")
    model, vocab, device, model_configs = init_scgpt_model()
    
    # 2. 
    task_list = get_all_expression_files()
    if not task_list:
        print("[ERROR] No valid tasks. Exit.")
        return
    
    # 3. 
    success_count = 0
    fail_count = 0
    print("\n[INFO] Start batch processing...")
    for task in task_list:
        if process_single_dataset(model, vocab, device, model_configs, task):
            success_count += 1
        else:
            fail_count += 1
    
    # 4. 
    print("\n" + "="*80)
    print("[INFO] Batch summary")
    print("="*80)
    print(f":{len(task_list)}")
    print(f":{success_count}")
    print(f":{fail_count}")
    print(f":{OUTPUT_ROOT}")
    print("[INFO] Full directed edges are exported (self-loop removed only).")
    print("="*80)

if __name__ == "__main__":
    main()
