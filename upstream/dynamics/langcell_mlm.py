"""Load a complete trained MLM for the LangCell rank-swap probe.

The official LangCell ``cell_bert`` checkpoint contains an encoder, not an MLM
head. It cannot supply gene-token likelihoods for this probe on its own. Supply
a compatible checkpoint whose encoder AND MLM head have been trained together;
never attach a newly initialized vocabulary projection to the cell encoder.
"""

import torch.nn as nn


def load_langcell_mlm(model_dir):
    from transformers import BertForMaskedLM

    model, info = BertForMaskedLM.from_pretrained(
        model_dir, output_loading_info=True, ignore_mismatched_sizes=False,
    )
    missing = set(info.get("missing_keys", []))
    # Hugging Face can omit aliases of tied parameters from a checkpoint.
    # Accept an alias only when its other name is loaded and truly shares storage.
    aliases = {
        "cls.predictions.decoder.weight": (
            "bert.embeddings.word_embeddings.weight",
            model.get_output_embeddings().weight,
            model.get_input_embeddings().weight,
        ),
        "cls.predictions.decoder.bias": (
            "cls.predictions.bias",
            model.cls.predictions.decoder.bias,
            model.cls.predictions.bias,
        ),
        "cls.predictions.bias": (
            "cls.predictions.decoder.bias",
            model.cls.predictions.bias,
            model.cls.predictions.decoder.bias,
        ),
    }
    original_missing = set(missing)
    for alias, (other_name, parameter, other_parameter) in aliases.items():
        if (alias in missing and other_name not in original_missing
                and parameter is other_parameter):
            missing.remove(alias)
    # Position/token-type ids are deterministic buffers, not trained weights.
    missing.difference_update({"bert.embeddings.position_ids", "bert.embeddings.token_type_ids"})
    mismatched = info.get("mismatched_keys", [])
    errors = info.get("error_msgs", [])
    if missing or mismatched or errors:
        raise RuntimeError(
            "LangCell rank-swap inference requires a complete trained BERT MLM "
            f"checkpoint: {model_dir}. Missing weights: {sorted(missing)}; "
            f"mismatched weights: {mismatched}; loader errors: {errors}. "
            "The official LangCell cell_bert is encoder-only. Supply a compatible "
            "checkpoint with a trained MLM head; no random head or automatic "
            "Geneformer-head substitution is allowed."
        )
    unexpected = info.get("unexpected_keys", [])
    print(f"[LangCell MLM] missing=[] mismatched=[] unexpected={unexpected}")
    return model


class LangCellModel(nn.Module):
    """Return pretrained token logits with the existing swap-probe interface."""

    def __init__(self, model_dir):
        super().__init__()
        self.mlm = load_langcell_mlm(model_dir)

    def forward(self, input_ids, attention_mask):
        return self.mlm(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True,
        ).logits
