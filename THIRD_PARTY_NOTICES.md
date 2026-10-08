# Third-party notices and license scope

The root MIT license applies to this project's original contributions. It does
not grant rights to third-party material or supersede its original terms.

## Bundled GENIE3 implementation

- File: `upstream/baselines/genie3/GENIE3.py`.
- Source: [official GENIE3 repository](https://github.com/vahuynh/GENIE3),
  [Python implementation](https://github.com/vahuynh/GENIE3/blob/master/GENIE3_python/GENIE3.py).
- The upstream repository identifies V. A. Huynh-Thu, A. Irrthum, L. Wehenkel,
  and P. Geurts as the algorithm's authors.
- No explicit license grant for this Python file was located in the inspected
  upstream repository on 2026-10-08. The file is therefore **excluded from this
  project's MIT grant**. Its presence in a public repository is not itself a
  license grant. Redistribution or reuse requires applicable permission from
  the original rights holders; this notice does not supply that permission.

## External software and pretrained models

scGPT, Geneformer, LangCell, scFoundation, scCello, scPRINT/scDataLoader,
DeepSEM, GRNFormer, regformer, Arboreto, and their dependencies are external
software. Install them under their respective upstream terms. Model weights,
gene vocabularies, and datasets must be obtained separately and retain their
own terms; they are not covered by this project's MIT license.

The PIDC Python implementation cites NetworkInference.jl and BEELINE as
algorithm references in its source. Such citations do not transfer ownership
of those external projects or relicense their code.

Preserve any third-party copyright and license notices when distributing
material originating from those projects.
