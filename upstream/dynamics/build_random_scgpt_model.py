#!/usr/bin/env python3
"""Xavier/Glorot-random scGPT builder used by the Fig. 6a seeded control.

Only model construction lives here. Evaluation parameters come from
``run_scgpt_gene_results.py`` via ``run_random_scgpt_seeded.py``.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
os.environ["KMP_WARNINGS"] = "off"

MODEL_DIR = "/mnt/10T/yzn/benchmark_GRN/pre_scgpt/scGPT/scgpt_human"
sys.path.insert(0, "/mnt/10T/yzn/benchmark_GRN/pre_scgpt/scGPT")
from scgpt.model import TransformerModel  # noqa: E402
from scgpt.tokenizer.gene_tokenizer import GeneVocab  # noqa: E402


def init_weights_xavier_normal(model):
    """Initialize model weights with Xavier/Glorot normal distribution."""

    def _init_fn(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.xavier_normal_(m.weight, gain=1.0)
        elif hasattr(m, "weight") and hasattr(m.weight, "data"):
            try:
                nn.init.xavier_normal_(m.weight, gain=1.0)
            except Exception:
                pass
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(_init_fn)
    print("  [INFO] Applied Xavier/Glorot normal initialization to all weights")
    return model


def build_model(model_dir, device, use_pretrained=True):
    with open(Path(model_dir) / "args.json") as f:
        cfg = json.load(f)

    vocab = GeneVocab.from_file(Path(model_dir) / "vocab.json")
    for t in ["<pad>", "<cls>", "<eoc>"]:
        if t not in vocab:
            vocab.append_token(t)

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=cfg["embsize"],
        nhead=cfg["nheads"],
        d_hid=cfg["d_hid"],
        nlayers=cfg["nlayers"],
        vocab=vocab,
        pad_value=cfg["pad_value"],
        n_input_bins=cfg.get("n_bins", 51),
        use_fast_transformer=cfg.get("fast_transformer", True),
    )

    if use_pretrained:
        ckpt = torch.load(Path(model_dir) / "best_model.pt", map_location="cpu")
        model.load_state_dict(ckpt, strict=False)
        print("  [INFO] Loaded pretrained weights")
    else:
        print("  [INFO] Using Xavier/Glorot normal initialization (random weights)")
        model = init_weights_xavier_normal(model)

    model.to(device)
    model.eval()
    if device.type == "cuda":
        model.half()
    return model, vocab
