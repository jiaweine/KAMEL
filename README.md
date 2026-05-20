# KAMEL: A Universal KAN-Augmented Multimodal Mixture-of-Experts Framework for Learning Heterogeneous and Imbalanced Tabular Data

This repository contains the code and datasets for the paper:

> **KAMEL: A Universal KAN-Augmented Multimodal Mixture-of-Experts Framework for Learning Heterogeneous and Imbalanced Tabular Data**  
> Jiawei Wang\*, Zebang Zhong\*, Runze Cai, Wangkai Ji, Hanwen Ning†  
> *Information Sciences* (\* Equal contribution, † Corresponding author)

## Overview

Tabular classification under class imbalance and data heterogeneity remains challenging: GBDTs lack end-to-end representation learning, while deep tabular models tend to exploit majority-class shortcuts and lack dedicated heterogeneity handling.

KAMEL (**KAN-A**ugmented **M**ultimodal **E**xpert **L**earning) addresses these two problems jointly through four coordinated components:

- **SSMT** (Same-Source Modality Transformation): maps each tabular record into three aligned views — original tabular features, a visual image (via IGTD), and a serialized text sequence.
- **UKAE** (Unified KAN-Augmented Encoder): encodes all three modalities into a shared latent space using a novel **Radial Basis Function KAN (RKAN)**, which decomposes each mapping into a smooth global component (SiLU) and a localized nonlinear component (RBF), providing stronger expressivity and cross-modal alignment than standard MLP encoders.
- **HMoEF** (Heterogeneous Mixture-of-Experts Fusion): fuses the tri-modal embeddings through structurally diverse expert groups (CrossBN, CrossLN, Synergy) with sparse top-*k* routing and temperature-annealed soft gating, enabling sample-adaptive prediction across heterogeneous data distributions.
- **DCIM** (Dynamic Curriculum Imbalance Mitigation): mitigates class imbalance from two perspectives — adaptive balanced sampling during representation learning, and curriculum-based focal loss calibration during classifier refinement.

## Repository Structure

```
KAMEL_clean/
├── kamel/                    # Main KAMEL package
│   ├── models/               # RKAN layers, HMoEF, KAMEL model
│   ├── data/                 # Preprocessor, SSMT transforms (IGTD + text serialization)
│   ├── training/             # Trainer, DCIM losses, train_utils
│   └── utils/                # Config, metrics, visualization
├── model_comparison/         # Baseline implementations
│   ├── algorithms/           # XGBoost, LightGBM, CatBoost, FT-Transformer, TabNet,
│   │                         # SAINT, TabM, NODE, GANDALF, TabR, TabPFN, TabICL, ...
│   ├── ml_experiment.py      # Baseline experiment runner
│   └── algorithm_configs.py  # Hyperparameter configs for all baselines
├── tabPFN/                   # TabPFN source (baseline dependency)
├── tabicl_src/               # TabICL source (baseline dependency)
├── data/                     # 25 benchmark datasets (CSV)
├── train_kamel.py            # KAMEL training entry point
└── run_ml_comparison.py      # Baseline comparison entry point
```

## Datasets

25 benchmark tabular classification datasets from the UCI repository and scikit-learn.

| Task | Datasets |
|------|----------|
| Binary (13) | Banknote Auth., Blood Transfusion, Breast Cancer Wisc., Diabetes, Haberman's Survival, Heart Disease, Hepatitis, Liver Disorders, Mammographic Mass, MONK's-1, Parkinsons, Spambase, Vertebral Column |
| Multiclass (12) | Authorship Attr., Car Evaluation, Cardiotocography, CMC, MFeat-Fourier, MFeat-Morphological, MFeat-Zernike, Nursery, Solar Flare, Vehicle Silhouettes, Wine Quality (Red), Yeast |

All datasets are available in the `data/` directory.

## Usage

### Train KAMEL

```bash
python train_kamel.py \
    --dataset banknote \
    --data_path data/banknote.csv \
    --model_size base \
    --kan_type fast_kan \
    --epochs 50 \
    --sampling_strategy curriculum \
    --loss_strategy focal \
    --device cuda \
    --seed 42
```

**Key arguments:**

| Argument | Choices | Default | Description |
|----------|---------|---------|-------------|
| `--dataset` | see script | required | Dataset name |
| `--data_path` | path | `None` | CSV file path (optional) |
| `--model_size` | `small` `base` `large` `xlarge` | `base` | Model scale |
| `--kan_type` | `fast_kan` `kan` `cheby_kan` `wave_kan` | `fast_kan` | KAN layer type |
| `--epochs` | int | `10` | Total training epochs |
| `--phase1_epochs` | int | `None` | Phase-1 (representation) epochs |
| `--sampling_strategy` | `curriculum` `balanced` `smote` | `curriculum` | Imbalance sampling |
| `--loss_strategy` | `ldam` `focal` `adaptive` `cross_entropy` | `ldam` | Imbalance loss |
| `--device` | `cuda` `cpu` | `cuda` | Training device |
| `--seed` | int | `42` | Random seed |

### Run Baseline Comparison

```bash
# Run all baselines on one dataset
python run_ml_comparison.py --dataset banknote --algorithm all

# Run a single baseline
python run_ml_comparison.py --dataset banknote --algorithm xgboost
```

## Requirements

Core KAMEL dependencies:
```
torch
numpy
pandas
scikit-learn
imbalanced-learn
optuna
```

Baseline dependencies are bundled in `tabPFN/` and `tabicl_src/`. For other baselines (XGBoost, LightGBM, CatBoost, FT-Transformer, TabNet, SAINT, TabM, NODE, GANDALF, TabR), install via `pip` as needed.
