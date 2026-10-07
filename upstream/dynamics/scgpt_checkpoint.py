"""Fail-closed checkpoint loading for scGPT's dynamic mlm_output probe."""

from collections.abc import Mapping
import json

import torch


# These heads are not called by the dynamic probe (CLS/MVC/DAB are disabled).
# Every other model parameter/buffer is required, including the expression decoder.
UNUSED_HEAD_PREFIXES = ("cls_decoder.", "mvc_decoder.", "grad_reverse_discriminator.")
REQUIRED_GROUPS = ("encoder.", "value_encoder.", "transformer_encoder.", "decoder.")
QKV_ALIASES = (
    (".self_attn.in_proj_weight", ".self_attn.Wqkv.weight"),
    (".self_attn.in_proj_bias", ".self_attn.Wqkv.bias"),
)


def load_scgpt_dynamics_checkpoint(model, checkpoint, *, checkpoint_path):
    """Validate all mlm_output weights before loading; return a JSON-safe report.

    Supports a bare state dict, common checkpoint wrappers, DataParallel's module.
    prefix, and packed FlashAttention/PyTorch QKV aliases. No reshaping, partial
    embedding expansion, or arbitrary key renaming is allowed.
    """
    state = checkpoint
    if isinstance(state, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            if isinstance(state.get(key), Mapping):
                state = state[key]
                break
    if not isinstance(state, Mapping) or not state:
        raise TypeError(f"{checkpoint_path}: expected a nonempty scGPT state dict")
    if not all(isinstance(key, str) for key in state):
        raise TypeError(f"{checkpoint_path}: state dict keys must be strings")
    state = dict(state)  # Do not mutate the caller's checkpoint.
    if all(key.startswith("module.") for key in state):
        state = {key[len("module."):]: value for key, value in state.items()}

    target = model.state_dict()
    absent_groups = [prefix for prefix in REQUIRED_GROUPS
                     if not any(key.startswith(prefix) for key in target)]
    if absent_groups:
        raise RuntimeError(f"Unsupported scGPT dynamic model: missing groups {absent_groups}")
    required = {key for key in target if not key.startswith(UNUSED_HEAD_PREFIXES)}
    aliases = {}
    conflicts = []
    consumed = set()
    for target_key in target:
        for native, flash in QKV_ALIASES:
            if target_key.endswith(native):
                source_key = target_key[:-len(native)] + flash
            elif target_key.endswith(flash):
                source_key = target_key[:-len(flash)] + native
            else:
                continue
            if source_key not in state:
                continue
            if target_key in state:
                if (not torch.is_tensor(state[target_key])
                        or not torch.is_tensor(state[source_key])
                        or not torch.equal(state[target_key], state[source_key])):
                    conflicts.append([target_key, source_key])
            else:
                state[target_key] = state[source_key]
                aliases[target_key] = source_key
            consumed.add(source_key)

    matched = {}
    mismatches = {}
    for key, expected in target.items():
        if key not in state:
            continue
        value = state[key]
        if not torch.is_tensor(value) or value.shape != expected.shape:
            mismatches[key] = {
                "checkpoint_shape": list(value.shape) if torch.is_tensor(value) else None,
                "model_shape": list(expected.shape),
            }
        else:
            matched[key] = value

    missing = sorted(set(target) - set(matched))
    critical_missing = sorted(required - set(matched))
    unexpected = sorted(set(state) - set(target) - consumed)
    critical_unexpected = [key for key in unexpected
                           if key.startswith(REQUIRED_GROUPS + ("bn.", "dsbn.", "batch_encoder."))]
    report = {
        "checkpoint": str(checkpoint_path),
        "loaded_keys": len(matched),
        "model_keys": len(target),
        "required_keys": len(required),
        "missing_keys": missing,
        "critical_missing_keys": critical_missing,
        "unexpected_keys": unexpected,
        "critical_unexpected_keys": critical_unexpected,
        "shape_mismatches": mismatches,
        "qkv_aliases": aliases,
        "conflicting_qkv_keys": conflicts,
        "status": "rejected" if critical_missing or critical_unexpected or conflicts else "validated",
    }
    # Print rather than warnings.warn: both entry points suppress Python warnings.
    print("[scGPT checkpoint] " + json.dumps(report, sort_keys=True), flush=True)
    if critical_missing or critical_unexpected or conflicts:
        raise RuntimeError(
            f"{checkpoint_path}: scGPT pretrained checkpoint rejected; "
            f"{len(critical_missing)} required keys missing/incompatible, "
            f"{len(critical_unexpected)} unexpected dynamic keys, "
            f"{len(conflicts)} conflicting QKV pairs. See checkpoint report. "
            "Use the matching model configuration/checkpoint/backend; "
            "do not continue with partially initialized pretrained weights."
        )

    result = model.load_state_dict(matched, strict=False)
    # Inspect the actual return value as well as the pre-load validation above.
    if set(result.missing_keys) != set(missing) or result.unexpected_keys:
        raise RuntimeError(
            f"{checkpoint_path}: load_state_dict returned unexpected results: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    report["status"] = "loaded"
    print("[scGPT checkpoint] All dynamic encoder and expression-decoder weights loaded.", flush=True)
    return report
