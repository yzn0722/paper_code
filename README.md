# paper_code

Code for benchmarking gene regulatory networks (GRNs), extracting foundation-model representations, evaluating pseudotime dynamics, and generating manuscript figures. This README is the central installation guide, workflow guide, and script index. Datasets, pretrained weights, result tables, figures, and manuscript files are obtained or generated separately.

## Contents

1. [Layout and execution order](#1-layout-and-execution-order)
2. [Installation](#2-installation)
3. [Inputs and configuration](#3-inputs-and-configuration)
4. [Network extraction and baselines](#4-network-extraction-and-baselines)
5. [GRN evaluation](#5-grn-evaluation)
6. [Pseudotime dynamics and propagation](#6-pseudotime-dynamics-and-propagation)
7. [Figures](#7-figures)
8. [Validation and reproducibility](#8-validation-and-reproducibility)
9. [License](#9-license)

## 1. Layout and execution order

```text
paper_code/
├── README.md                 Central documentation
├── LICENSE                   License for original contributions
├── THIRD_PARTY_NOTICES.md     Third-party scope and attribution
├── requirements*.txt         Dependency profiles
├── environment.yml           General Conda environment
├── environments/             Observed server package inventory
├── upstream/
│   ├── network/               Model representations and GRN export
│   ├── baselines/             Expression and learned GRN baselines
│   ├── evaluation/            EPR and AUPR evaluation
│   └── dynamics/              Dynamics, direction metrics and propagation
├── plots/                    Main and supplementary figures
└── tests/                    CPU regression fixtures
```

Run commands from the repository root. Prepare the environment and external inputs, extract model networks or run selected baselines, then evaluate and plot them. Dynamics is a separate branch: generate predictions/trajectories, calculate direction or propagation scores, and plot those results. Scripts are individual entry points; there is no single command that recreates every manuscript panel.

## 2. Installation

### 2.1 General environment

The general environment targets **Python 3.10**:

```sh
conda env create -f environment.yml
conda activate paper-code-analysis
python -m pip check
```

Alternatively, create a Python 3.10 virtual environment:

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
```

| Profile | Purpose |
| --- | --- |
| [requirements.txt](requirements.txt) | General evaluation, plotting, expression baselines and PIDC. |
| [requirements-tests.txt](requirements-tests.txt) | General dependencies plus PyTorch, Transformers and safetensors for CPU regression tests. |
| [requirements-singlecell.txt](requirements-singlecell.txt) | Shared Scanpy/AnnData, HDF5/loom, UMAP, gene lookup and Transformer utilities; install alongside general requirements in a new utility environment. |
| [environment.yml](environment.yml) | Conda configuration for general analysis. |
| [scgpt-server-observed.txt](environments/scgpt-server-observed.txt) | Existing Python 3.8.19 server's direct-package inventory, recorded on 2026-10-08. |

For shared single-cell utilities:

```sh
python -m pip install -r requirements.txt -r requirements-singlecell.txt
```

Profiles pin direct dependencies, not a complete transitive lock or a universal environment for all models. The Python 3.10 PIDC profile uses Numba 0.59.1 with NumPy 1.25.2 rather than the older server's Numba pin. Import/distribution names differ: `sklearn` → `scikit-learn`, `community` → `python-louvain`, `umap` → `umap-learn`, `yaml` → `PyYAML`. `openpyxl` supplies the Fig. 2 Excel engine.

### 2.2 Model environments

Install matching model source, dependencies and checkpoints from the release used for the analysis. Use separate environments when requirements conflict, and choose a PyTorch/CUDA build compatible with the release and GPU driver. Do not overwrite a working model environment with general utility pins. Use the [official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/) for platform-specific GPU/CPU wheels.

| Model | Additional source and resources |
| --- | --- |
| scGPT | Matching `scgpt` source, `args.json`, `vocab.json` and checkpoint. Observed server metadata reports `0.1.6`, which alone does not identify locally modified source. Backend selection must pass checkpoint coverage checks. |
| Geneformer | Matching `geneformer` source where imported, Transformers, datasets, token/mapping dictionaries and checkpoint; vocabulary/configuration must match the model generation. |
| LangCell | Transformers and a complete trained MLM checkpoint including its output head. The rank-swap probe rejects encoder-only checkpoints. |
| scFoundation | Original dependencies and source tree with `model/load.py` and `model/pretrainmodels/`, plus gene list/checkpoint. `load` and `pretrainmodels` are source modules, not generic PyPI packages. |
| scCello | Matching `sccello` and `sc_foundation_evals` source, including upstream `model_prototype_contrastive` modules, original dependencies and checkpoint. |
| scPRINT | Compatible `scprint`, `scdataloader`, original dependencies and checkpoint; these packages were absent from the inspected scGPT server environment. |
| DeepSEM | Separate upstream checkout/requirements; configure the wrapper's `DEEPSEM_MAIN_PATH`. |
| GRNFormer / regformer | Separate upstream source, matching requirements/configuration, preprocessing resources and checkpoint; configure source roots and runner Python paths. |

Bash launchers require Linux, macOS or WSL. The server inventory is not a complete Conda export and does not capture CUDA/drivers/compiler versions or model-source changes. Fresh installation of every model/profile has not been validated.

### 2.3 GRNBoost and R/PPCOR

GRNBoost2 requires `arboreto`. The inspected server used `arboreto==0.1.6`, `dask==2023.5.0`, `distributed==2023.5.0`, and `pandas==1.5.0`. Use a suitable separate environment, match Dask/distributed versions, and check compatibility before combining that historical stack with newer pandas.

PPCOR requires R, `Rscript` on PATH (or a configured runner path), and CRAN `ppcor`. Python requirements do not install R packages:

```sh
Rscript -e "install.packages('ppcor', repos='https://cloud.r-project.org')"
Rscript -e "library(ppcor); packageVersion('ppcor')"
```

Use a writable R library and record R/package versions. The script prepends a historical user-library path but also searches standard R libraries.

## 3. Inputs and configuration

| Input | Structure / role |
| --- | --- |
| Expression CSV | Genes in rows, cells in columns, gene identifiers in the first column. Preserve identifiers/order when matching pseudotime and trajectories. |
| Ground-truth CSV | Directed `Gene1`, `Gene2` edges; commonly stored in `CHIP`, `Non_CHIP`, `STRING` subdirectories. |
| Prediction TSV | `Gene1`, `Gene2`, `EdgeWeight`; Gene1 is source/regulator in the query-to-key convention. |
| Pseudotime CSV | Cell-aligned pseudotime, commonly `<dataset>/PseudoTime.csv`; use the selected reader's supported columns. |
| Model resources | Checkpoint, configuration, vocabulary/gene list, dictionaries and external source tree as required by the model. |
| Plot inputs | Evaluation summaries, curves, gene-result CSVs, saved trajectories/manifests; Fig. 2 also expects an external Excel summary. |

Common expression names are `*_chip_matched-ExpressionData.csv` (CHIP) and `*_processed-ExpressionData.csv` (other sources); corresponding ground truths use `*-network.csv`. Dataset selections vary by runner.

Some scripts retain historical absolute paths. Set their configuration constants or available path arguments before execution. Inspect `--help` for argparse entry points in the correct environment; inspect the top configuration block for scripts without a CLI. Use new output directories when changing inputs, checkpoints or protocols, and retain manifests.

## 4. Network extraction and baselines

### 4.1 Model network extraction

Token and contextual hidden representations produce gene cosine networks; attention scripts export directed gene-pair weights. Availability differs by model.

| Model | Token cosine | Hidden cosine | Attention |
| --- | --- | --- | --- |
| scGPT | [token](upstream/network/extract_scgpt_token.py) | [hidden](upstream/network/extract_scgpt_hidden.py) | [attention](upstream/network/extract_scgpt_attention.py) |
| Geneformer | [token](upstream/network/extract_geneformer_token.py) | [hidden](upstream/network/extract_geneformer_hidden.py) | [attention](upstream/network/extract_geneformer_attention.py) |
| LangCell | [token](upstream/network/extract_langcell_token.py) | [hidden](upstream/network/extract_langcell_hidden.py) | [attention](upstream/network/extract_langcell_attention.py) |
| scCello | [token](upstream/network/extract_sccello_token.py) | [hidden](upstream/network/extract_sccello_hidden.py) | [attention](upstream/network/extract_sccello_attention.py) |
| scFoundation | [token](upstream/network/extract_scfoundation_token.py) | [hidden](upstream/network/extract_scfoundation_hidden.py) | — |
| scPRINT | — | — | [attention](upstream/network/extract_scprint_attention.py) |

Additional exports: [export_scgpt_attention_heads.py](upstream/network/export_scgpt_attention_heads.py) for individual heads, and [export_attention_tsv.py](upstream/network/export_attention_tsv.py) for attention export. [utils_heads.py](upstream/network/utils_heads.py) is a shared helper, not an experiment entry point.

### 4.2 Baselines

| Baseline | Entry point / supporting source |
| --- | --- |
| Pearson, Spearman, mutual information | [run_expression_baselines.py](upstream/baselines/run_expression_baselines.py) |
| PIDC / Spearman partial correlation (PPCOR) | [run_beeline_baselines.py](upstream/baselines/classical/run_beeline_baselines.py), with [pidc.py](upstream/baselines/classical/pidc.py) and [run_ppcor.R](upstream/baselines/classical/run_ppcor.R). |
| GENIE3 | [run_genie3_batch.py](upstream/baselines/genie3/run_genie3_batch.py), using bundled [GENIE3.py](upstream/baselines/genie3/GENIE3.py); see the license exclusion below. |
| GRNBoost2 | [run_grnboost.py](upstream/baselines/grnboost/run_grnboost.py) |
| DeepSEM | [run_deepsem_batch.py](upstream/baselines/deepsem/run_deepsem_batch.py) |
| GRNFormer | [run_grnformer_baseline.py](upstream/baselines/grnformer/run_grnformer_baseline.py) and [run_grnformer_beeline.sh](upstream/baselines/grnformer/run_grnformer_beeline.sh) |
| regformer token embeddings | [prepare_beeline_h5ad.py](upstream/baselines/regformer/prepare_beeline_h5ad.py), then [run_grn_token_emb.py](upstream/baselines/regformer/run_grn_token_emb.py); launcher: [run_token_emb_pipeline.sh](upstream/baselines/regformer/run_token_emb_pipeline.sh). |

Example:

```sh
python upstream/baselines/run_expression_baselines.py \
  --gt-root /path/input_process --expr-source STRING --gt-types STRING \
  --datasets hESC --methods pearson_abs spearman_abs mi \
  --outdir outputs/expression_baselines
```

MI preserves paired cell observations when reshaping one-hot arrays. DeepSEM reads actual upstream `GRN_inference_result.tsv` output and fails on missing/malformed output instead of substituting a random network. The classical [__init__.py](upstream/baselines/classical/__init__.py) is a package marker.

## 5. GRN evaluation

| Metric | Entry point |
| --- | --- |
| Early precision ratio (EPR) | [evaluate_epr.py](upstream/evaluation/evaluate_epr.py) |
| AUPR and AUPR/random-baseline ratio | [evaluate_aupr.py](upstream/evaluation/evaluate_aupr.py) |

Match prediction and ground-truth directory/file naming to the evaluator configuration; configure model/dataset selections as needed:

```sh
python upstream/evaluation/evaluate_epr.py \
  --pred_root /path/predictions --true_root /path/input_process \
  --output outputs/epr_results.csv
python upstream/evaluation/evaluate_aupr.py \
  --pred_root /path/predictions --true_root /path/input_process \
  --output outputs/aupr_results.csv
```

These examples retain default TF filtering. EPR ranks by absolute weight and uses `K=min(eligible predictions, ground-truth edges)` after filtering/deduplication. It truncates the sorted list at K; boundary ties are not jointly admitted and missing predictions are not padded to the ground-truth edge count. Check the selected evaluator's candidate-space and score conventions when comparing different runners.

## 6. Pseudotime dynamics and propagation

### 6.1 Entry points

| Analysis | Entry point |
| --- | --- |
| Formal scGPT gene results | [run_scgpt_gene_results.py](upstream/dynamics/run_scgpt_gene_results.py) |
| Supplementary mDC scGPT run | [run_scgpt_supplementary_mdc.py](upstream/dynamics/run_scgpt_supplementary_mdc.py): append the sixth dataset using the same validated loaded model and protocol as the five-dataset reference. |
| Matched random/pretrained scGPT | [run_random_scgpt_seeded.py](upstream/dynamics/run_random_scgpt_seeded.py); builder: [build_random_scgpt_model.py](upstream/dynamics/build_random_scgpt_model.py). |
| scGPT initial-state sensitivity | [run_scgpt_initial_state.py](upstream/dynamics/run_scgpt_initial_state.py) |
| Other model probes | [Geneformer](upstream/dynamics/run_geneformer_balanced.py), [LangCell](upstream/dynamics/run_langcell_balanced.py), [scFoundation](upstream/dynamics/run_scfoundation_balanced.py), [scPRINT](upstream/dynamics/run_scprint_pseudotime.py). |
| Unified entry | [run_multimodel_pseudotime.py](upstream/dynamics/run_multimodel_pseudotime.py) |
| Trajectories and GRN validation | [run_dynamic_grn_validation.py](upstream/dynamics/run_dynamic_grn_validation.py) |
| Weighted GRN propagation | [run_weighted_grn_propagation.py](upstream/dynamics/run_weighted_grn_propagation.py) |
| Fig. 6c query-to-key recomputation | [run_fig06c_query_to_key.py](upstream/dynamics/run_fig06c_query_to_key.py) |
| Cross-dataset propagation | [run_cross_dataset_grn_density.py](upstream/dynamics/run_cross_dataset_grn_density.py), [shell launcher](upstream/dynamics/run_cross_dataset_grn_density.sh). |

Shared modules: [direction_metrics.py](upstream/dynamics/direction_metrics.py) for scoring, [pseudotime_utils.py](upstream/dynamics/pseudotime_utils.py) for finite/aligned pseudotime, [scgpt_checkpoint.py](upstream/dynamics/scgpt_checkpoint.py) for critical checkpoint coverage, and [langcell_mlm.py](upstream/dynamics/langcell_mlm.py) for trained MLM loading. Keep these modules with their runners when deploying. LangCell uses `LANGCELL_MLM_MODEL_DIR` for a compatible complete checkpoint and rejects an untrained output head.

### 6.2 Metric and formal scGPT protocol

Direction scoring selects the top 30% of mapped genes by absolute observed early-to-late change. `EPS_DIR=0.001` excludes near-zero observed directions after selection; near-zero predictions count as incorrect for a defined observed direction. BA averages the recalls of observed classes present: a single-class set uses that recall, and an empty set returns NaN. Compatibility filenames such as `accuracy_curves.json` may contain BA; inspect metric metadata.

The matched formal scGPT experiment uses native per-cell binning of mapped genes, no log1p, 16 iterations, and `next = 0.9 * old + 0.1 * mlm_output`. Binning occurs at initialization; subsequent states remain continuous, with CLS/padding frozen. Refinement occurs per cell before averaging, and required checkpoint coverage is validated. Other entry points retain their own settings/manifests; matching metric names alone does not establish identical protocols.

### 6.3 Fig. 6a: matched random and pretrained runs

Configure the formal evaluator's data/model paths, then use new directories:

```sh
python upstream/dynamics/run_random_scgpt_seeded.py \
  --outdir outputs/fig06a_current/random \
  --pretrained-outdir outputs/fig06a_current/pretrained
python plots/fig06a_pretrained_vs_random_scgpt.py \
  --random-results-dir outputs/fig06a_current/random \
  --weight-results-dir outputs/fig06a_current/pretrained \
  --vocab-path /path/scgpt_human/vocab.json \
  --output-pdf outputs/fig06a_current/fig06a.pdf
```

Random seeds default to 1–10. Each dataset resets preprocessing to `20261008 + dataset index` across conditions, holding quantile ties and observed references fixed while model initialization varies. Results use `seed_01`–`seed_10` subdirectories. The plot checks manifests, input/protocol equality and CSV hashes before recomputing BA. Random error bars are sample SD over ten seeds; pretrained is a single run. Reuse requires matching protocol and hashes; legacy results are rejected.

### 6.4 Fig. 5c/d/e: current scGPT results

```sh
python upstream/dynamics/run_scgpt_initial_state.py \
  --pretrained-dir outputs/fig06a_current/pretrained \
  --outdir outputs/fig05_current/initial_states
python plots/fig05_scgpt_panels.py \
  --pretrained-dir outputs/fig06a_current/pretrained \
  --curves-json /path/curves_all_models.json \
  --trajectory-dir outputs/fig05_current/initial_states \
  --outdir outputs/fig05_current/figures
```

The initial-state output directory must be new. All cells in each starting group are used. The Early trajectory must exactly match the formal reference in panel c; Intermediate/Late starts need not match it. Panels c/d display 11 iterations by default (`--c-iterations 16` for the full run). Panel e plots all mapped genes at iteration 16; BA uses mapped top 30%. Other models' supplied curves are preserved, not independently validated. Historical LangCell curves from a random output head require recomputation using a trained compatible MLM head before scientific reuse.

The panels share the configured font/page dimensions and export PDF/SVG, PNG and source/provenance files. A single trajectory has no replicate-based error bars; gene-wise correlation P values are descriptive.

### 6.4.1 Supplementary Figs. 3/4: six matched current scGPT datasets

Reuse the five pretrained outputs above and run only mDC into a new directory:

```sh
python upstream/dynamics/run_scgpt_supplementary_mdc.py \
  --reference-dir outputs/fig06a_current/pretrained \
  --outdir outputs/supplementary_current/mdc
python plots/supp05_balanced_convergence_six_datasets.py \
  --curves-json /path/previous_curves_all_models.json \
  --pretrained-dir outputs/fig06a_current/pretrained \
  --supplemental-dir outputs/supplementary_current/mdc \
  --outdir outputs/supplementary_current/fig3
python plots/supp03_gene_change_scatter_six_datasets.py \
  --pretrained-dir outputs/fig06a_current/pretrained \
  --supplemental-dir outputs/supplementary_current/mdc \
  --outdir outputs/supplementary_current/fig4
```

If the five-dataset run predates the optional trajectory recorder, provide its checksum-matching archived evaluator with `--reference-evaluator /path/reference/run_scgpt_gene_results.py`. The runner accepts only that recorder addition with identical inference AST; other source/checkpoint/model mismatches stop the run. The appended mDC preprocessing seed is 20261013, preserving the five established reference seeds. Configure expression/pseudotime/model paths in the formal evaluator before generating a new reference.

Supplementary Fig. 3 keeps the original first 11 displayed iterations (`--iterations 16` shows the complete run) and exports all 16 steps. It replaces all six scGPT curves, with other supplied model curves preserved. Supplementary Fig. 4 displays all mapped genes at step 16. Both validate CSV checksums and the shared model/protocol. The supplied SI template uses `SL/SL3.pdf` for convergence and `SL/SL4.pdf` for gene-change scatter; historical script numbers are retained for compatibility.

### 6.5 Fig. 6c/d: trajectories and rewired controls

The trajectory generator performs preflight by default; `--execute` runs inference. Supply expression/pseudotime roots, scGPT source/model directories and a new output directory. Defaults are EMA retention 0.9, 32 iterations, up to 16 cells per group, batch size 4 and seed 0; these differ from the formal 16-iteration/all-cell Fig. 5 recipe. It exports group mean trajectories and protocol metadata.

Recompute Fig. 6c from saved hESC trajectories, then explicitly plot without vertical error bars:

```sh
python upstream/dynamics/run_fig06c_query_to_key.py \
  --trajectory-dir /path/hESC_trajectory --benchmark-root /path/benchmark_GRN \
  --outdir outputs/fig06c/query_to_key
python plots/fig06c_plot_density_hESC.py \
  --json-dir outputs/fig06c/query_to_key/raw \
  --outdir outputs/fig06c/query_to_key --error none
```

The recomputation wrapper requests descriptive lag SD in its plot call; the second command selects the no-error-bar presentation. No scGPT inference is rerun by this recomputation. Primary orientation is `query_to_key_primary` (Gene1 → Gene2), with reverse `key_to_query_sensitivity`. The calculator in dynamics and [fig06c_grn_propagation_hESC.py](plots/fig06c_grn_propagation_hESC.py) use the same implementation. Do not relabel legacy reverse-direction reports.

The hESC recipe uses 1k/5k/10k/20k/30k edges, 200 rewired networks per representation/density, and median Spearman over the first eight lags. Base seed is 20260903 with representation offsets 0/1009/2018. Observed and null inputs come from the same reports. Grey display pools 3 × 200 null draws per density; individual reports retain representation-specific comparisons. Eight lags from one trajectory are not independent replicates: lag spread is descriptive source data, while shading shows the pooled null 5th–95th percentile.

## 7. Figures

Use outputs from the corresponding analyses and configure input/output locations in the script or CLI.

| Panel | Entry / purpose |
| --- | --- |
| Fig. 2 | [fig02_grn_performance_heatmap.py](plots/fig02_grn_performance_heatmap.py): performance heatmap from external Excel summary. |
| Fig. 3a–c | [fig03a_c_topology_radar_hESC.py](plots/fig03a_c_topology_radar_hESC.py): topology radar plots. |
| Fig. 3d | [fig03d_modularity_hESC.py](plots/fig03d_modularity_hESC.py): modularity. |
| Fig. 3e/f | [fig03e_f_scgpt_networks_hESC.py](plots/fig03e_f_scgpt_networks_hESC.py): scGPT network/community views. |
| Fig. 3g | [fig03g_average_degree_hESC.py](plots/fig03g_average_degree_hESC.py): average degree. |
| Fig. 3h/i | [fig03h_i_degree_ccdf_hESC.py](plots/fig03h_i_degree_ccdf_hESC.py): degree CCDFs. |
| Fig. 4a | [fig04a_edge_precision_hESC.py](plots/fig04a_edge_precision_hESC.py): precision with matched TF→target reference. |
| Fig. 4b | [fig04b_tf_f1_hESC.py](plots/fig04b_tf_f1_hESC.py): TF-level overlap/F1. |
| Fig. 4c | [fig04c_tf_overlap_hESC.py](plots/fig04c_tf_overlap_hESC.py): TF overlap. |
| Fig. 4d | [fig04d_consensus_tf_network_hESC.py](plots/fig04d_consensus_tf_network_hESC.py): consensus TF network. |
| Fig. 4e | [fig04e_tf_degree_rank_hESC.py](plots/fig04e_tf_degree_rank_hESC.py): TF out-degree rank. |
| Fig. 4f | [fig04f_tf_enrichment_hESC.py](plots/fig04f_tf_enrichment_hESC.py): TF enrichment. |
| Fig. 5b | [fig05b_scgpt_umap_mHSC-L.py](plots/fig05b_scgpt_umap_mHSC-L.py): joint observed/generated-cell UMAP. |
| Fig. 5c/d/e | Combined [fig05_scgpt_panels.py](plots/fig05_scgpt_panels.py); individual entries [c](plots/fig05c_balanced_convergence_mHSC-L.py), [d](plots/fig05d_initial_state_mHSC-L.py), [e](plots/fig05e_gene_change_scatter_mHSC-L.py). See Section 6.4. |
| Fig. 5f | [fig05f_balanced_accuracy_five_datasets.py](plots/fig05f_balanced_accuracy_five_datasets.py): supplied five-dataset BA summary, excluding mDC; 14/16 pt DejaVu Sans and the same page/axes height as Fig. 5c/d/e; does not recompute predictions. |
| Fig. 6a | [fig06a_pretrained_vs_random_scgpt.py](plots/fig06a_pretrained_vs_random_scgpt.py): matched ten-seed control. |
| Fig. 6c | [fig06c_plot_density_hESC.py](plots/fig06c_plot_density_hESC.py): propagation versus density. |
| Fig. 6d | [fig06d_cross_dataset_grn_propagation.py](plots/fig06d_cross_dataset_grn_propagation.py): cross-dataset coupling. |
| Supplementary UMAP | [supp01_scgpt_umap_six_datasets.py](plots/supp01_scgpt_umap_six_datasets.py) |
| Supplementary attention heads | [supp02_scgpt_attention_heads_hESC.py](plots/supp02_scgpt_attention_heads_hESC.py) |
| Supplementary Fig. 4 gene changes | [supp03_gene_change_scatter_six_datasets.py](plots/supp03_gene_change_scatter_six_datasets.py): checksum-validated current native-binning results; all mapped genes, 16 steps, EMA 0.9. Supply `--pretrained-dir`, `--supplemental-dir` for the mDC run, and `--outdir`. |
| Supplementary Fig. 3 convergence | [supp05_balanced_convergence_six_datasets.py](plots/supp05_balanced_convergence_six_datasets.py): six current native-binning scGPT curves, with other supplied model curves preserved; provide `--curves-json`, `--pretrained-dir`, `--supplemental-dir`, and `--outdir`. Defaults to the original 11 displayed steps; exports all 16 steps as source data. |

Shared palettes: [fig2_palette.py](plots/fig2_palette.py), [fig3_palette.py](plots/fig3_palette.py), [fig4_palette.py](plots/fig4_palette.py). Supplementary UMAP uses native per-cell binning and EMA retention 0.9, resetting seed 42 per dataset. Each panel fits its own scaler/PCA/UMAP; coordinates across panels are not one shared embedding. Configure `ROOT`, `SCGPT_REPO`, `MODEL_DIR` before execution.

## 8. Validation and reproducibility

In a separate Python 3.10 environment:

```sh
python -m pip install -r requirements-tests.txt
python -m pip check
python -m unittest discover -s tests -v
```

For CPU-only PyTorch, install the matching CPU wheel first using the official version instructions. Select an individual group with `python -m unittest discover -s tests -p test_scgpt_checkpoint.py -v`.

| Test | Coverage |
| --- | --- |
| [test_baseline_logic.py](tests/test_baseline_logic.py) | Paired-cell MI, real DeepSEM outputs, trained LangCell head. |
| [test_direction_metrics.py](tests/test_direction_metrics.py) | BA, undefined directions, runner exports and figure inputs. |
| [test_pseudotime_validation.py](tests/test_pseudotime_validation.py) | Finite/aligned pseudotime and grouping. |
| [test_scgpt_checkpoint.py](tests/test_scgpt_checkpoint.py) | Required coverage, QKV aliases, shapes and rejection before mutation. |
| [test_scgpt_dynamics_protocol.py](tests/test_scgpt_dynamics_protocol.py) | Mapped binning, EMA, frozen inputs and export. |
| [test_fig04a_candidates.py](tests/test_fig04a_candidates.py) | Matched curve/reference candidate spaces. |
| [test_fig05_current_protocol.py](tests/test_fig05_current_protocol.py) | Current source/protocol and initial-state trajectories. |
| [test_supplementary_current_protocol.py](tests/test_supplementary_current_protocol.py) | Six current scGPT curves, preserved other models, rejected missing/mismatched mDC and inconsistent endpoints. |
| [test_fig06a_seeded_protocol.py](tests/test_fig06a_seeded_protocol.py) | Condition inputs, manifests and safe reuse. |
| [test_fig06c_direction.py](tests/test_fig06c_direction.py) | Orientation and matching observed/null sources. |

Tests use synthetic fixtures and explicit external-model stubs where appropriate. They do not validate real FlashAttention kernels, pretrained inference, all model environments or manuscript values. Scientific reproduction additionally requires external inputs and matching model/protocol versions.

Retain input/checkpoint/vocabulary/source hashes, preprocessing, EMA, iteration counts, dataset selections, seeds, backend and metric metadata with each run. Data, weights, figures, archives and caches are excluded from the public code tree.

## 9. License

Original contributions use the [MIT license](LICENSE). External software, datasets and pretrained weights retain their own terms. Bundled [GENIE3.py](upstream/baselines/genie3/GENIE3.py) is excluded from this project's MIT grant because an explicit upstream license was not located. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and scope.
