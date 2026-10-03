# `upstream/dynamics/run_geneformer_balanced.py`

File-level notes generated from the current server source by static inspection.

## Purpose (source docstring)

Geneformer swap=8 | 6个数据集 | 输出纯JSON收敛曲线 | 无图无多余输出

## Dependencies

Imports found in the source (standard library and external modules): `json`, `numpy`, `os`, `pandas`, `pathlib`, `pickle`, `time`, `torch`, `transformers`, `warnings`.

## Defined interfaces

`normalize_symbol`, `is_ensembl_id`, `read_pt_file`, `load_geneformer_dicts`, `build_symbol_to_ensembl`, `direction_accuracy_top_genes`, `balanced_accuracy_top_genes`, `positions_dict_from_seq`, `iterative_swap_sampling_geneformer`, `run_dataset`, `main`

## Invocation

Run `python upstream/dynamics/run_geneformer_balanced.py` from the repository root after supplying the external inputs and configuring paths. Review argument defaults before running.

## Referenced paths and file names

These literals may designate inputs, outputs, or templates. Consult their surrounding source code for their role; these files are not included.

- `/mnt/10T/yzn/benchmark_GRN/PseudoTime/hESC/PseudoTime.csv`
- `/mnt/10T/yzn/benchmark_GRN/PseudoTime/hHep/PseudoTime.csv`
- `/mnt/10T/yzn/benchmark_GRN/PseudoTime/mDC/PseudoTime.csv`
- `/mnt/10T/yzn/benchmark_GRN/PseudoTime/mHSC-E/PseudoTime.csv`
- `/mnt/10T/yzn/benchmark_GRN/PseudoTime/mHSC-GM/PseudoTime.csv`
- `/mnt/10T/yzn/benchmark_GRN/PseudoTime/mHSC-L/PseudoTime.csv`
- `/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/hESC_chip_matched-ExpressionData.csv`
- `/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/hHep_chip_matched-ExpressionData.csv`
- `/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/mDC_chip_matched-ExpressionData.csv`
- `/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/mHSC-E_chip_matched-ExpressionData.csv`
- `/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/mHSC-GM_chip_matched-ExpressionData.csv`
- `/mnt/10T/yzn/benchmark_GRN/input_process/CHIP/mHSC-L_chip_matched-ExpressionData.csv`
- `/mnt/10T/yzn/benchmark_GRN/model/weights/Geneformer/default/6L`
- `/mnt/10T/yzn/benchmark_GRN/model/weights/Geneformer/dicts`
- `/mnt/10T/yzn/scGRN-Bench/FBplot/fig4/balanced_convergence_work/geneformer_balanced_accuracy_curves.json`
- `gene_name_id_dict.pkl`
- `token_dictionary.pkl`

## Data and runtime requirements

Datasets, checkpoints, generated figures, result tables, caches, and logs are excluded. Supply required inputs separately. Server-specific paths may need adjustment. Documentation is based on static source inspection; model execution and end-to-end reproduction have not been tested.

Source: [run_geneformer_balanced.py](run_geneformer_balanced.py)
