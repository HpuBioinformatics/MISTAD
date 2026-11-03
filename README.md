# MISTAD

**MISTAD** is a pure-regression deep learning framework for detecting topologically associating domain (TAD) boundaries from Hi-C contact maps. The current implementation focuses on the human GM12878 cell line at 25 kb resolution (GRCh38/hg38).

## Overview

MISTAD uses five Hi-C-derived features—`insulation_100kb`, `insulation_150kb`, `directionality_index`, `up_mean`, and `down_mean`—extracted from 21-bin sliding windows. A dilated residual 1D convolutional neural network with squeeze-and-excitation (SE) blocks predicts boundary-associated scores by regressing Gaussian-smoothed labels (σ = 2.5 bins). Predicted boundary candidates are clustered and further filtered using a one-sided Wilcoxon rank-sum test (*p* < 0.05).

The workflow comprises six stages:

- **S1–S2 — Hi-C processing:** Convert `.hic` files to KR-normalized contact matrices and extract the five features.
- **S3 — Boundary filtering (optional):** Filter consensus boundary candidates using protein peak annotations. The resulting label BED file is provided in this repository.
- **S4 — Dataset preparation:** Construct labeled 21-bin windows for model training.
- **S5 — Training:** Train the regression model using the prepared windows.
- **S6 — Prediction:** Predict TAD boundaries using a trained checkpoint, window tensors, and KR-normalized contact matrices.

Protein peak data are used only for optional boundary-label regeneration (S3), **not for model inference**. The training tensors and KR matrices are not distributed and must be generated before running S5 or S6, respectively.

## Requirements

The following software is needed to run MISTAD:

- **Python:** 3.9 or later (Python 3.10 recommended).
- **Python packages:** PyTorch, NumPy, pandas, SciPy, and scikit-learn.
- **GPU support:** A CUDA-enabled PyTorch installation is recommended for model training.
- **Java:** Version 8 or later, required only for Hi-C extraction with Juicer Tools (S1).

Install the Python dependencies in your preferred environment:

```bash
python -m pip install torch numpy pandas scipy scikit-learn
```

Choose a PyTorch build compatible with your local CUDA installation if using a GPU. Exact dependency versions have not yet been pinned; recording them in a requirements file would improve reproducibility.

## Quick Start

The following commands outline the path from raw Hi-C data to predicted TAD boundaries. Run them from the repository root after installing the dependencies.

**Before running S1:** Obtain the GM12878 Hi-C file (accession `4DNFI1UEG1HD`) and update the `HIC_FILE` constant in `data/extract_hic_to_npy.py` to point to the downloaded `.hic` file. S3 can be skipped because the filtered boundary BED file is already included.

```bash
# S1: Extract 25 kb KR-normalized Hi-C matrices.
python data/extract_hic_to_npy.py

# S2: Extract the five Hi-C features.
python data/run_extract_feature.py

# S4: Generate training tensors using the supplied boundary labels.
python feature/generate_datasets.py

# S5: Train the final regression model.
python training/MISTAD_train.py --mode single --profile reg_base --seeds 1001

# S6: Prepare the contact matrices needed for prediction on chr20–22.
mkdir -p prediction_outputs/hic
for chrom in 20 21 22; do
    cp "data/raw/GM12878_25kb/chr${chrom}_25kb_hicKR.npy" \
       "prediction_outputs/hic/chr${chrom}_dense.npy"
done

# Predict TAD boundaries on the default test chromosomes (chr20–22).
python prediction/MISTAD_predict.py \
    --training-root outputs/model_reg_v1 \
    --hic-dir prediction_outputs/hic \
    --profiles reg_base --seed 1001
```

This workflow requires downloading the raw Hi-C data. The full details, inputs, and outputs of each step are described below.

## Project Structure

```text
MISTAD/
├── data/                                          # Raw Hi-C data processing
│   ├── extract_hic_to_npy.py                      # S1: .hic -> 25 kb KR matrices
│   ├── extract_5_feature.py                       # S2: five-feature extraction
│   ├── run_extract_feature.py                     # S2: feature extraction entry point
│   ├── juicer_tools.jar
│   └── raw/
│       ├── GM12878_25kb/                          # S1 output: chr{N}_25kb_hicKR.npy
│       └── TAD_boundary/
│           ├── consensus_tads.bed                 # Reference consensus TADs
│           └── filtered_boundaries.filtered.bed   # S3 labels: 3,884 boundaries (included)
├── feature/
│   ├── generate_datasets.py                       # S4: labeled training windows
│   ├── generate_predict_datasets.py               # S6: unlabeled prediction windows
│   └── GM12878_25kb_feature/                      # S2 output: chr{N}_hic_features.csv
├── training/
│   └── MISTAD_train.py                            # S5: pure-regression model training
├── prediction/
│   └── MISTAD_predict.py                          # S6: prediction and Wilcoxon filtering
├── outputs/
│   └── model_reg_v1/                               # S5 output: pretrained weights (~16 MB, included)
│       ├── canonical_initialization/
│       └── runs/reg_base/seed_1001/{sig}/best_model.pth
├── filter_boundaries.py                           # S3: optional protein-based label filtering
└── datasets/
    └── ws21/                                       # S4 output: chr{N}_ws21.npz
```

The filtered boundary labels and pretrained model weights are included in the repository. Raw KR matrices, feature CSV files, and training tensors must be generated locally.

## Pipeline

### S1. Hi-C matrix extraction

Convert GM12878 Hi-C data to 25 kb, KR-normalized dense matrices using Juicer Tools.

- **Input:** Juicer `.hic` file (4D Nucleome accession `4DNFI1UEG1HD`).
- **Configuration:** Set `HIC_FILE` in `data/extract_hic_to_npy.py`.
- **Output:** `data/raw/GM12878_25kb/chr{N}_25kb_hicKR.npy`.

```bash
python data/extract_hic_to_npy.py
```

Skip this step if compatible KR matrices have already been prepared.

### S2. Feature extraction

Extract the five Hi-C-derived features from the KR-normalized matrices.

- **Input:** `data/raw/GM12878_25kb/chr{N}_25kb_hicKR.npy`.
- **Output:** `feature/GM12878_25kb_feature/chr{N}_hic_features.csv`, together with a global robust scaler (`gm12878_training_global_robust_scaler.npz`).

```bash
python data/run_extract_feature.py
```

Feature CSV files are generated locally and are not included in the repository.

### S3. Boundary confidence filtering (optional)

This step uses consensus boundary candidates and protein peak BED files to identify higher-confidence training labels. It is **not needed** when using the supplied labels.

- **Input:** A tab-separated candidate BED file containing `chrom`, `start`, `end`, `support_tool_count`, and `support_tool_names`, plus protein peak files under `data/proteins/{protein}.bed`.
- **Filtering criteria:** `VERY_HIGH` requires support from at least three callers and overlaps with CTCF, RAD21, and SMC3. `HIGH` requires either (i) at least two callers with CTCF and RAD21 or SMC3, or (ii) at least three callers with RAD21 and SMC3.
- **Script outputs:** `{out-prefix}.annotated.tsv` and `{out-prefix}.filtered.tsv`.
- **Provided training labels:** `data/raw/TAD_boundary/filtered_boundaries.filtered.bed` (3,884 boundaries).

```bash
python filter_boundaries.py \
    --candidates /path/to/candidate_boundaries.bed \
    --out-prefix data/raw/TAD_boundary/filtered_boundaries
```

The exact candidate-boundary input used to create the supplied labels is not distributed. The relationship between the script's `.filtered.tsv` output and the supplied `.filtered.bed` file should be documented before attempting to regenerate S3 independently.

### S4. Training dataset generation

Construct 21-bin windows from the feature CSV files and the supplied boundary labels.

- **Input:** Feature CSV files from S2 and `filtered_boundaries.filtered.bed`.
- **Output:** `datasets/ws21/chr{N}_ws21.npz`.
- **Stored arrays:** `features` of shape `[N, 21, 5]` and `soft_labels` of shape `[N, 21]`, together with hard labels and sample indices.
- **Soft labels:** `exp(-d² / (2σ²))`, where *d* is the distance in bins to the nearest boundary and σ = 2.5 bins. Bins with soft-label values above 0.99 are treated as hard positives; the regression objective uses soft labels.
- **Chromosome split:** Training on chr1–17 (excluding chr7 and chr10), validation on chr18–19, and testing on chr20–22.

```bash
python feature/generate_datasets.py
```

### S5. Model training

Train the pure-regression model using the tensors generated in S4.

```bash
# Train the final model (seed 1001).
python training/MISTAD_train.py --mode single --profile reg_base --seeds 1001

# Optional short run to check the training pipeline.
python training/MISTAD_train.py --mode single --profile reg_base --epochs 2 --no-test
```

Training outputs are saved under:

```text
outputs/model_reg_v1/runs/{profile}/seed_1001/{config_signature}/
```

The run directory contains `best_model.pth`, `config.json`, `summary.json`, `history.csv`, and `test_predictions.npz`. Shared initialization information is stored in `canonical_initialization/`.

Useful options include `--epochs` (default: 80), `--batch-size` (default: 256), `--data-base`, `--output-root`, `--strict-deterministic`, and `--force`.

**Note:** Values passed to `--exclude-train-chrs` must include the `chr` prefix (for example, `chr2,chr3`).

### S6. TAD boundary prediction

Prediction uses (1) a checkpoint generated in S5 or the pretrained checkpoint included in `outputs/model_reg_v1/`, (2) 21-bin window tensors, and (3) KR-normalized dense matrices renamed as `chr{N}_dense.npy`.

By default, the script predicts on **chr20, chr21, and chr22**. Make sure the corresponding KR matrices are available in `prediction_outputs/hic/` before running:

```bash
python prediction/MISTAD_predict.py \
    --training-root outputs/model_reg_v1 \
    --hic-dir prediction_outputs/hic \
    --profiles reg_base --seed 1001
```

For prediction with newly generated, unlabeled windows:

```bash
python feature/generate_predict_datasets.py

python prediction/MISTAD_predict.py \
    --training-root outputs/model_reg_v1 \
    --hic-dir prediction_outputs/hic \
    --data-dir datasets/predict_ws21/ws21_25kb_GM12878 \
    --profiles reg_base --seed 1001
```

Prediction files are written under `prediction_outputs/TADs_reg/{profile}/`, including:

- `25KB_TAD/chr{N}.txt`: Final TAD regions as `start_bin end_bin` pairs at 25 kb resolution (with the start bin shifted by +1).
- `predictions/predictions.txt`: Per-bin prediction records, including `cls_prob`, `reg_prob`, thresholds, and candidate flags.
- `predictions/boundaries_wilcoxon.txt`: Candidate boundary statistics, including Wilcoxon *p*-values, effect estimates, and selection flags.

Additional options include `--chroms`, `--pvalue-cutoff` (default: 0.05), `--resolution-kb` (default: 25), and `--validate-only`.

## Results

The following validation metrics are reported for the final `reg_base` model (seed 1001, epoch 22) in `summary.json`:

- **AUPRC:** 0.575
- **F1 score:** 0.635
- **Precision:** 0.561
- **Recall:** 0.730
- **Soft-label Pearson correlation (*r*):** 0.634

Corresponding manuscript tables and figures will be linked when available.

## Data Availability

The analysis uses human **GM12878** data aligned to **GRCh38/hg38**.

**Hi-C data**

- **4D Nucleome Data Portal:** `4DNFI1UEG1HD` — processed Hi-C contact data used to prepare 25 kb KR-normalized matrices (S1 and S6).

**Protein peak datasets (ENCODE)**

The following peak BED files are used only when regenerating boundary labels in S3:

- **CTCF:** `ENCFF827JRI`
- **RAD21:** `ENCFF940IMA`
- **SMC3:** `ENCFF837YJA`
- **H3K27ac:** `ENCFF367KIF`
- **H3K4me3:** `ENCFF188SZS`
- **H3K4me1:** `ENCFF321BVG`
- **H3K36me3:** `ENCFF268HMO`
- **H3K9me3:** `ENCFF682WIQ`
- **POLR2A:** `ENCFF886PSD`
- **EZH2:** Accession to be added.

Additional bigWig tracks were used for visualization only and are not needed to run the pipeline: H3K27ac (`ENCFF087YCU`), H3K4me3 (`ENCFF480KNX`), H3K9me3 (`ENCFF533NIQ`), CTCF (`ENCFF680XUD`), and POLR2A (`ENCFF532WJA`).

## Reproducibility Notes

- The repository includes the S3 label BED file with **3,884 boundaries** and pretrained model weights under `outputs/model_reg_v1/` (~16 MB), but does not distribute raw KR matrices, extracted feature CSV files, or training tensors.
- The supplied label BED allows S3 to be skipped; S1, S2, and S4 must still be completed to prepare training data.
- Each training run records a `code_sha256`, a configuration signature, and a canonical-initialization hash in `config.json`.
- The documented workflow follows S2 → S4 → S5 → S6, with S1 required when starting from a raw `.hic` file. Execution should be verified in the target environment before publication.
- Regenerating S1 requires the original `.hic` input and Java. Regenerating S3 requires protein peak BEDs and the original candidate-boundary file, which is not currently provided.
- Exact dependency versions and the link between the S3 `.filtered.tsv` output and the provided `.filtered.bed` labels remain to be documented.

## Citation

Citation details will be added when the associated manuscript or preprint is available.
