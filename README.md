<div align="center">

# KAMEL

### KAN-Augmented Multimodal Expert Learning

**A universal KAN-augmented multimodal mixture-of-experts framework for learning heterogeneous and imbalanced tabular data**

Jiawei Wang<sup>†</sup> · Zebang Zhong<sup>†</sup> · Runze Cai · Wangkai Ji · Hanwen Ning<sup>*</sup>

[![Paper](https://img.shields.io/badge/Paper-Information%20Sciences-4F46E5?style=flat-square)](https://doi.org/10.1016/j.ins.2026.123925)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.ins.2026.123925-0EA5E9?style=flat-square)](https://doi.org/10.1016/j.ins.2026.123925)
[![Volume](https://img.shields.io/badge/Information%20Sciences-757%20(2026)%20123925-0891B2?style=flat-square)](https://doi.org/10.1016/j.ins.2026.123925)
[![Framework](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-25%20datasets-16A34A?style=flat-square)](#experimental-results)

[**Paper**](https://doi.org/10.1016/j.ins.2026.123925) · [**Method**](#method) · [**Quick Start**](#quick-start) · [**Results**](#experimental-results) · [**Citation**](#citation)

<sub><sup>†</sup> Equal contribution &nbsp;&nbsp; <sup>*</sup> Corresponding author</sub>

</div>

<br/>

<p align="center">
  <a href="https://ars.els-cdn.com/content/image/1-s2.0-S002002552600856X-gr1_lrg.jpg">
    <img src="https://ars.els-cdn.com/content/image/1-s2.0-S002002552600856X-gr1_lrg.jpg" width="100%" alt="Figure 1: The KAMEL framework" />
  </a>
</p>

<p align="center"><em><b>Figure 1.</b> The KAMEL framework. Click the figure to open the publisher's large-resolution original.</em></p>

> [!IMPORTANT]
> **KAMEL** jointly addresses two coupled challenges in tabular learning: **distributional heterogeneity** and **class imbalance**. It transforms each tabular record into aligned tabular, visual, and textual views, aligns them through a unified **RKAN-based encoder**, adaptively fuses them with structurally heterogeneous experts, and mitigates imbalance through a two-stage dynamic curriculum.

---

## ✨ Highlights

- **Same-source multimodal representation.** SSMT generates tabular, visual, and textual views from the same record without requiring naturally multimodal data.
- **Unified RKAN encoding.** UKAE employs the proposed **Radial Basis Function KAN (RKAN)**, combining a smooth SiLU backbone with localized RBF responses.
- **Heterogeneity-aware expert fusion.** HMoEF organizes experts into **CrossBN**, **CrossLN**, and **Synergy** groups and uses noisy exploration, sparse top-*k* routing, and temperature-annealed soft gating.
- **Dynamic imbalance mitigation.** DCIM separates representation-oriented balanced learning from calibration-oriented refinement toward the empirical distribution.
- **Strong empirical performance.** KAMEL achieves the best or tied-best macro-F1 on all **25** evaluated benchmark datasets in the paper.

## Why KAMEL?

| Challenge | Limitation | KAMEL response |
|---|---|---|
| **Class imbalance** | Majority-class gradients can dominate representation learning and bias decision boundaries. | **DCIM** coordinates balanced representation learning with decision-boundary calibration. |
| **Distributional heterogeneity** | One global predictor may fail to preserve subgroup-specific decision patterns. | **HMoEF** routes samples to structurally different expert functions. |
| **Complex nonlinear interactions** | Conventional MLP encoders can be limited in localized nonlinear modeling and cross-view alignment. | **UKAE + RKAN** models global smooth structure and local RBF variation jointly. |
| **Single-view tabular representation** | A single feature geometry may hide complementary spatial or semantic structure. | **SSMT** creates aligned tabular, visual, and textual views from the same source. |

---

# Method

KAMEL coordinates four components: **SSMT → UKAE/RKAN → HMoEF**, with **DCIM** integrated into the training process.

## 1. SSMT — Same-Source Modality Transformation

For every raw tabular sample, SSMT constructs three aligned views.

**Tabular view.** Numerical features are normalized, categorical variables are encoded, and missing values are imputed according to feature type.

**Visual view.** KAMEL uses **Image Generator for Tabular Data (IGTD)**. Feature-to-pixel locations are optimized so statistically dependent features are placed close to one another. The paper builds feature-distance rankings from pairwise Pearson correlations and iteratively improves the feature-to-pixel mapping before rendering each sample as a 2-D image.

**Textual view.** Human-readable feature names and the **raw feature values** are serialized into ordered feature–relation–value sequences such as `feature is value ;`. Continuous values are formatted to a fixed precision before tokenization, truncation/padding, and attention-mask construction.

> The three modalities are **same-source views** of one underlying tabular sample rather than independent data sources.

## 2. UKAE — Unified KAN-Augmented Encoder

UKAE maps the tabular, visual, and textual modalities into a shared latent space. Its key nonlinear component is the paper's proposed **Radial Basis Function KAN (RKAN)**.

For output channel *j*, RKAN is defined as:

```math
\phi_j(\mathbf{x})
= \sum_{i=1}^{d_{in}}
\left[
W_b^{(i,j)}\,\sigma(x_i)
+ W_s^{(i,j)}
\sum_{k=1}^{G} W_{\mathrm{rbf}}^{(i,j,k)}
\exp\!\left(-\frac{(x_i-g_{k,i})^2}{h^2}\right)
\right],
\qquad \sigma=\mathrm{SiLU}.
```

The two terms have complementary roles:

- **SiLU component:** captures the global and smooth nonlinear mapping.
- **RBF component:** captures localized nonlinear variation around reference grid points.

In the paper's RKAN configuration, **8 RBF grid points** are uniformly distributed over **[-2, 2]**, and SiLU is used as the base activation.

The three encoder paths are:

- **Tabular:** stacked RKAN layers with LayerNorm and dropout → `e_tab`
- **Image:** convolution + BatchNorm + ReLU + max pooling, followed by RKAN → `e_img`
- **Text:** token embedding + sinusoidal positional encoding + convolution + adaptive max pooling, followed by RKAN → `e_txt`

All three representations are projected to the same latent dimension before fusion.

> [!NOTE]
> **Paper/code naming.** The published method proposes **RKAN**. The current repository CLI exposes implementation switches named `kan`, `fast_kan`, `cheby_kan`, and `wave_kan`. In the paper, **FastKAN is evaluated as a separate KAN variant from the proposed RKAN**, so this README intentionally does not equate the CLI label `fast_kan` with RKAN.

## 3. HMoEF — Heterogeneous Mixture-of-Experts Fusion

<p align="center">
  <a href="https://ars.els-cdn.com/content/image/1-s2.0-S002002552600856X-gr2_lrg.jpg">
    <img src="https://ars.els-cdn.com/content/image/1-s2.0-S002002552600856X-gr2_lrg.jpg" width="100%" alt="Figure 2: HMoEF module" />
  </a>
</p>

<p align="center"><em><b>Figure 2.</b> HMoEF. Click the figure to open the publisher's large-resolution original.</em></p>

HMoEF receives the concatenated tri-modal embedding `e = [e_tab; e_img; e_txt]` and partitions its experts into three functional groups:

| Expert group | Normalization | Nonlinearities | Main role |
|---|---|---|---|
| **CrossBN** | BatchNorm | ReLU / GELU / SiLU | Cross-modal interactions under relatively stable feature statistics. |
| **CrossLN** | LayerNorm | LeakyReLU / ELU / PReLU | Cross-modal interactions under stronger instance-level distribution shifts. |
| **Synergy** | BatchNorm | Mish | Deeper modeling of higher-order three-modal dependencies. |

Routing follows four stages:

1. **Logit generation** by a lightweight MLP.
2. **Gaussian-noise injection** during training for exploration and to reduce premature routing collapse.
3. **Sparse top-*k* selection**, activating only the selected experts for each sample.
4. **Temperature-annealed soft weighting**, moving from smoother routing toward sharper specialization as training proceeds.

The reported setup activates **top-2** experts from **3 CrossLN + 3 CrossBN + 1 Synergy** experts. HMoEF additionally uses **load-balancing** and **diversity** regularization.

The fused representation is:

```math
\mathbf{h}=\sum_{j=1}^{K} w_j\,E_j(\mathbf{e}).
```

## 4. DCIM — Dynamic Curriculum Imbalance Mitigation

<p align="center">
  <a href="https://ars.els-cdn.com/content/image/1-s2.0-S002002552600856X-gr3_lrg.jpg">
    <img src="https://ars.els-cdn.com/content/image/1-s2.0-S002002552600856X-gr3_lrg.jpg" width="100%" alt="Figure 3: DCIM workflow" />
  </a>
</p>

<p align="center"><em><b>Figure 3.</b> DCIM. Click the figure to open the publisher's large-resolution original.</em></p>

DCIM treats imbalance as **both a representation-learning problem and a decision-boundary calibration problem**.

**Phase 1 — representation-oriented learning.** From epoch 1 to `T1`, mini-batches are constructed with approximately balanced class exposure. Minority classes are oversampled toward a target determined by the majority-class count and an oversampling intensity coefficient.

```math
\tilde n_c=\lfloor \alpha\,n_{\max}\rfloor.
```

**Phase 2 — calibration-oriented refinement.** From `T1 + 1` to the final epoch `T`, the sampling distribution is gradually annealed from the balanced target toward the original class frequencies:

```math
\tilde n_c(t)
=\tilde n_c
+\frac{t-T_1}{T-T_1}\left(n_c-\tilde n_c\right).
```

DCIM also introduces a class-frequency-dependent margin:

```math
\Delta_y=\frac{m_{\max}}{n_y^{1/4}}.
```

Rarer classes therefore receive a larger margin. The margin-adjusted probability is combined with focal weighting:

```math
\hat p_y
=\frac{\exp(z_y-\Delta_y)}
{\exp(z_y-\Delta_y)+\sum_{j\ne y}\exp(z_j)},
\qquad
\mathcal{L}_{cls}=-(1-\hat p_y)^\gamma\log(\hat p_y).
```

The overall objective includes the classification loss together with HMoEF load-balancing and diversity regularizers:

```math
\mathcal{L}
=\mathcal{L}_{cls}
+\lambda_{load}\mathcal{L}_{load}
+\lambda_{div}\mathcal{L}_{div}.
```

---

# Experimental Results

The paper evaluates KAMEL on **25 public tabular classification datasets** — **13 binary** and **12 multiclass** tasks — against **12 representative baselines** spanning tree ensembles, deep tabular models, and tabular foundation models.

<table>
<tr>
<td align="center"><strong>25 / 25</strong><br/>best or tied-best F1</td>
<td align="center"><strong>+2.79%</strong><br/>avg. relative gain, binary</td>
<td align="center"><strong>+1.36%</strong><br/>avg. relative gain, multiclass</td>
<td align="center"><strong>5</strong><br/>datasets at 100% F1</td>
</tr>
</table>

### Main findings

- **Best or tied-best F1-score on all 25 datasets.**
- Average relative improvement over the strongest competing baseline: **2.79%** on binary tasks and **1.36%** on multiclass tasks.
- Relative gains reach **4.19% on Heart Disease** and **5.13% on Solar Flare**.
- KAMEL reaches **100.0% F1** on Banknote, MONK's-1, Car Evaluation, Cardiotocography, and Nursery.
- KAMEL is the only compared method to reach **100.0% F1 on Nursery**.

### Ablation highlights

| Ablation / replacement | Avg. relative F1 change |
|---|---:|
| UKAE/RKAN → standard MLP | **−11.62%** |
| Remove tabular modality | **−10.58%** |
| Remove visual modality | **−6.84%** |
| Remove textual modality | **−9.81%** |
| HMoEF → concatenation | **−7.10%** |
| HMoEF → late fusion | **−6.23%** |
| DCIM → single-stage cross-entropy | **−7.59%** |
| DCIM → single-stage focal loss | **−7.82%** |

These ablations support the central design of KAMEL: its gains arise from the **coordination** of multimodal representation, RKAN encoding, heterogeneous expert routing, and curriculum-based imbalance mitigation.

## Benchmark suite

**Binary (13):** Banknote, Blood Transfusion, Breast Cancer, Diabetes, Haberman, Heart Disease, Hepatitis, Liver Disease, Mammographic, MONK's-1, Parkinsons, Spambase, Vertebral Column.

**Multiclass (12):** Authorship, Car Evaluation, Cardiotocography, CMC, MFeat-Four, MFeat-Morph, MFeat-Zern, Nursery, Solar Flare, Vehicle, Wine Quality, Yeast.

**Baselines (12):** XGBoost, LightGBM, CatBoost, NODE, FT-Transformer, TabNet, SAINT, TabM, TabR, TabPFN, TabICL, and GANDALF.

---

# Repository

```text
KAMEL/
├── kamel/
│   ├── data/
│   │   ├── dataset_configs.py     # Dataset-specific preprocessing metadata
│   │   ├── preprocessor.py        # Tabular preprocessing and splits
│   │   └── transforms.py          # IGTD + text transformation
│   ├── models/
│   │   ├── kamel_model.py         # End-to-end KAMEL model
│   │   ├── kan_encoders.py        # Tabular / image / text KAN encoders
│   │   ├── kan_layers.py          # KAN-family layers
│   │   └── moe_fusion.py          # Heterogeneous MoE and sparse routing
│   ├── training/
│   │   ├── imbalance_handler.py   # Dynamic curriculum sampling
│   │   ├── losses.py              # Imbalance-aware objectives
│   │   ├── trainer.py
│   │   └── train_utils.py
│   └── utils/                     # Metrics, config, visualization
├── model_comparison/              # Baseline implementations / runners
├── tabPFN/                        # TabPFN baseline source
├── tabicl_src/                    # TabICL baseline source
├── assets/paper/                  # Local paper-figure fallbacks
├── train_kamel.py                 # KAMEL training entry point
└── run_ml_comparison.py           # Baseline comparison entry point
```

## Installation

```bash
git clone https://github.com/jiaweine/KAMEL.git
cd KAMEL

pip install \
  torch numpy pandas scipy \
  scikit-learn imbalanced-learn \
  transformers optuna pyyaml \
  matplotlib seaborn
```

Baseline models may require additional model-specific dependencies.

---

# Quick Start

### Train with the current repository CLI

```bash
python train_kamel.py \
  --dataset banknote \
  --data_path /path/to/banknote.csv \
  --model_size base \
  --epochs 50 \
  --sampling_strategy curriculum \
  --loss_strategy focal \
  --device cuda \
  --seed 42
```

> [!TIP]
> This command documents the **current repository interface**. For the exact experimental settings reported in the paper, refer to **Section 5.1 and Appendix D**, including dataset-specific hyperparameters selected by Bayesian optimization.

### Key CLI arguments

| Argument | Choices / type | Current default | Description |
|---|---|---:|---|
| `--dataset` | supported alias | required | Dataset configuration. |
| `--data_path` | path | `None` | Explicit local dataset path. |
| `--model_size` | `small`, `base`, `large`, `xlarge` | `base` | Model-scale preset. |
| `--kan_type` | `kan`, `fast_kan`, `cheby_kan`, `wave_kan` | `fast_kan` | Current code implementation switch; see the paper/code note above. |
| `--epochs` | int | `10` | Total training epochs. |
| `--phase1_epochs` | int | `None` | Override the first curriculum phase length. |
| `--sampling_strategy` | `curriculum`, `balanced`, `smote` | `curriculum` | Sampling strategy. |
| `--loss_strategy` | `ldam`, `focal`, `adaptive`, `cross_entropy` | `ldam` | Current code loss switch. |
| `--device` | `cuda`, `cpu` | `cuda` | Training device. |
| `--seed` | int | `42` | Random seed. |

If CUDA is requested but unavailable, the training script falls back to CPU.

### Dataset files

The repository contains dataset configuration metadata under `kamel/data/`, but does not currently vendor a complete root-level `data/` directory containing every benchmark dataset. Provide the local path when required:

```bash
python train_kamel.py \
  --dataset spambase \
  --data_path /path/to/spambase.data \
  --device cuda
```

### Baseline comparisons

```bash
# All configured baselines
python run_ml_comparison.py --dataset banknote --algorithm all

# One baseline
python run_ml_comparison.py --dataset banknote --algorithm xgboost
```

---

# Paper

> **KAMEL: A universal KAN-augmented multimodal mixture-of-experts framework for learning heterogeneous and imbalanced tabular data**  
> Jiawei Wang, Zebang Zhong, Runze Cai, Wangkai Ji, Hanwen Ning  
> *Information Sciences*, **757** (2026), 123925  
> DOI: [10.1016/j.ins.2026.123925](https://doi.org/10.1016/j.ins.2026.123925)

Accepted **19 July 2026** · Available online **20 July 2026**.

# Citation

```bibtex
@article{wang2026kamel,
  title   = {KAMEL: A universal KAN-augmented multimodal mixture-of-experts framework for learning heterogeneous and imbalanced tabular data},
  author  = {Wang, Jiawei and Zhong, Zebang and Cai, Runze and Ji, Wangkai and Ning, Hanwen},
  journal = {Information Sciences},
  volume  = {757},
  pages   = {123925},
  year    = {2026},
  doi     = {10.1016/j.ins.2026.123925}
}
```

## Acknowledgement

This repository includes or interfaces with multiple open-source tabular-learning baselines. Please cite the original publications of the individual baseline methods used in your experiments.

---

<div align="center">

**SSMT · UKAE / RKAN · HMoEF · DCIM**

<sub>Learning expressive and robust tabular representations under heterogeneity and class imbalance.</sub>

</div>
