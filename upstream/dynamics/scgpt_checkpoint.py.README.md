# `upstream/dynamics/scgpt_checkpoint.py`

Shared loader for the scGPT expression dynamics probe. Dependencies: Python standard library and PyTorch. Keep this helper beside either dynamic entry point when deploying.

Call `load_scgpt_dynamics_checkpoint(model, checkpoint, checkpoint_path=path)` with the constructed model and the result of `torch.load`. It returns a JSON-safe report. It requires all model state except explicitly unused `cls_decoder`, `mvc_decoder` and `grad_reverse_discriminator` heads. Gene/value encoder, transformer and expression decoder groups must exist. The dynamic probe must keep CLS/MVC/DAB disabled, as the current entry points do.

Supports bare state dicts or dictionaries containing `model_state_dict`, `state_dict`, or `model`, and strips a uniform `module.` prefix. Only packed QKV aliases (`Wqkv.weight/bias` versus `in_proj_weight/bias`) are translated. Shapes must match; conflicting aliases and extra dynamic-backbone keys fail. Input checkpoints are not mutated. Missing/incompatible required state is rejected before model weights are changed. The actual `load_state_dict` return is checked after loading.

Reports use `print(..., flush=True)` because the callers suppress Python warnings. Missing/extra/mismatched unused heads are reported but do not prevent expression-only evaluation. Name/shape validation cannot prove backend numerical equivalence, input preprocessing correctness, or historical result validity.

Regression tests: `python -m unittest discover -s tests -p test_scgpt_checkpoint.py -v`. Full execution requires scGPT, checkpoint/config/vocabulary files and external datasets.
