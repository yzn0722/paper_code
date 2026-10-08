# `upstream/dynamics/langcell_mlm.py`

Shared strict LangCell masked-language-model loader for both dynamic runners. Requires PyTorch and Hugging Face Transformers. `LangCellModel(model_dir)` loads a compatible trained BERT encoder plus MLM prediction head with `BertForMaskedLM`; its forward output is the model's vocabulary logits.

Missing encoder/head weights, shape mismatches, or load errors fail explicitly. Tied decoder aliases are accepted only when the loaded counterpart actually shares the same parameter. The official encoder-only LangCell cell checkpoint has no MLM head and is insufficient for this swap-based dynamic probe. It now fails instead of constructing an untrained random Linear layer. Provide a genuinely trained compatible full checkpoint; the code neither trains a head nor substitutes a Geneformer head. Both runners use this same implementation.
