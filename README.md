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

<p align="center">
  <img src="assets/paper/kamel_framework.webp" width="100%" alt="Figure 1: The KAMEL framework" />
</p>

<p align="center"><em>Figure 1. The KAMEL framework from the published paper: SSMT generates aligned multimodal views, UKAE maps them into a shared embedding space, HMoEF performs heterogeneous expert fusion, and DCIM mitigates class imbalance throughout training.</em></p>

> [!IMPORTANT]
> **KAMEL** jointly targets two coupled difficulties in tabular learning: **distributional heterogeneity** and **class imbalance**. A single tabular record is transformed into aligned tabular, visual, and textual views; the views are encoded by a unified **RKAN-based** encoder; a structurally heterogeneous MoE adaptively routes each sample; and a two-stage dynamic curriculum calibrates both representations and decision boundaries.

---

## Highlights

- **Same-source multimodal learning.** SSMT derives tabular, visual, and textual representations from the *same* record, without requiring naturally multimodal data.
- **Unified RKAN encoding.** UKAE uses the proposed **Radial Basis Function KAN (RKAN)** to combine smooth global nonlinear structure with localized RBF responses.
- **Heterogeneity-aware fusion.** HMoEF partitions experts into **CrossBN**, **CrossLN**, and **Synergy** groups and combines them with noisy exploration, sparse top-*k* routing, and temperature-annealed soft gating.
- **Imbalance-aware curriculum.** DCIM first learns approximately class-balanced representations, then gradually restores the empirical class distribution while applying class-dependent margins and focal weighting.
- **Strong benchmark performance.** The paper reports the best or tied-best macro-F1 on all **25** evaluated datasets.

## Why KAMEL?

| Challenge | What goes wrong | KAMEL response |
|---|---|---|
| **Class imbalance** | Majority-class gradients can dominate representation learning and bias the final decision boundary. | **DCIM** separates representation-oriented balancing from calibration-oriented refinement. |
| **Distributional heterogeneity** | A single global predictor may fail to preserve subgroup-specific decision patterns. | **HMoEF** routes different samples toward structurally different expert functions. |
| **Complex nonlinear interactions** | Standard MLP encoders may be insufficient for cross-view alignment and localized nonlinear effects. | **UKAE + RKAN** combines a smooth SiLU backbone with learnable local RBF responses. |
| **Single-view tabular representation** | One feature geometry may hide complementary spatial or semantic structure. | **SSMT** constructs aligned tabular, visual, and textual views from the same source. |

---

# Method

KAMEL is organized around four coordinated components: **SSMT → UKAE/RKAN → HMoEF**, with **DCIM** integrated into the training process.

## 1. SSMT — Same-Source Modality Transformation

For each raw tabular record, SSMT constructs three aligned representations:

**Tabular view.** Numerical features are normalized, categorical variables are encoded, and missing values are imputed according to feature type. This produces the structured representation used by the tabular encoder.

**Visual view.** KAMEL uses **Image Generator for Tabular Data (IGTD)**. Feature-to-pixel positions are optimized so statistically dependent features are placed close to one another. The paper forms feature-distance rankings from pairwise Pearson correlations and iteratively improves the feature-to-pixel mapping before rendering each sample as a 2-D image.

**Textual view.** Human-readable feature names and the **raw feature values** are serialized as ordered feature–relation–value sequences (for example, `feature is value ;`). Continuous values are formatted to a fixed precision, then the sequence is tokenized, truncated/padded, and paired with an attention mask.

> The three views are **same-source**: they contain complementary structural, spatial, and semantic descriptions of one underlying tabular sample rather than independent modalities.

## 2. UKAE — Unified KAN-Augmented Encoder

UKAE maps all three modalities into a shared latent dimension \(D\). Its core nonlinear layer is the paper's proposed **Radial Basis Function KAN (RKAN)**.

For output channel \(j\), RKAN combines two complementary terms:

$$
\phi_j(\mathbf{x}) = \sum_i \left[
W_b^{(i,j)}\,\mathrm{SiLU}(x_i)
+ W_s^{(i,j)} \sum_k W_{\mathrm{rbf}}^{(i,j,k)}
\exp\!\left(-\frac{(x_i-g_{k,i})^2}{h^2}\right)
\right].
$$

The **SiLU term** models a stable global and smooth mapping, while the **RBF term** captures localized nonlinear variation around reference grid points. In the reported RKAN setup, the paper uses **8 RBF grid points** uniformly spaced over **[-2, 2]**, with the corresponding kernel width and SiLU as the base activation.

The three encoding paths are modality-specific before entering RKAN layers:

- **Tabular:** stacked RKAN layers with LayerNorm and dropout → \(\mathbf{e}_{tab}\)
- **Image:** convolution + BatchNorm + ReLU + max pooling, followed by RKAN → \(\mathbf{e}_{img}\)
- **Text:** token embedding + sinusoidal positional encoding + convolution + adaptive max pooling, followed by RKAN → \(\mathbf{e}_{txt}\)

All three outputs share the same embedding dimension and are passed to HMoEF.

> [!NOTE]
> **Paper/code naming:** the published method uses **RKAN**. The current repository CLI exposes implementation switches named `kan`, `fast_kan`, `cheby_kan`, and `wave_kan`. In the paper, **FastKAN is evaluated as a separate KAN variant from the proposed RKAN**. Therefore, this README does **not** equate the CLI label `fast_kan` with the paper's RKAN.

## 3. HMoEF — Heterogeneous Mixture-of-Experts Fusion

<p align="center">
  <img src="assets/paper/hmoef.webp" width="100%" alt="Figure 2: HMoEF module" />
</p>

<p align="center"><em>Figure 2. HMoEF routes the concatenated tri-modal embedding through structurally diverse expert groups and a sparse top-k gate, with diversity and load-balancing regularization.</em></p>

HMoEF receives the concatenated tri-modal embedding
\(\mathbf{e}=[\mathbf{e}_{tab};\mathbf{e}_{img};\mathbf{e}_{txt}]\) and partitions its experts into three functional groups:

| Expert group | Normalization | Nonlinearities | Intended behavior |
|---|---|---|---|
| **CrossBN** | BatchNorm | ReLU / GELU / SiLU | Cross-modal interactions under relatively stable feature statistics. |
| **CrossLN** | LayerNorm | LeakyReLU / ELU / PReLU | Robust cross-modal modeling under instance-level distribution shifts. |
| **Synergy** | BatchNorm | Mish | Deeper expert for higher-order three-modal dependencies. |

Routing follows four stages:

1. **Logit generation** with a lightweight MLP on the concatenated embedding.
2. **Gaussian noise injection** during training to encourage exploration and reduce premature routing collapse.
3. **Sparse top-*k* selection**, activating only the most relevant experts for each sample.
4. **Temperature-annealed soft weighting**, beginning with smoother routing and progressively sharpening expert specialization.

The reported setup activates the **top 2** experts from **3 CrossLN + 3 CrossBN + 1 Synergy** experts. HMoEF also uses **load-balancing** and **diversity** regularization to discourage expert under-utilization and redundant routing patterns.

## 4. DCIM — Dynamic Curriculum Imbalance Mitigation

<p align="center">
  <img src="assets/paper/dcim.webp" width="100%" alt="Figure 3: DCIM workflow" />
</p>

<p align="center"><em>Figure 3. DCIM combines an epoch-gated two-phase curriculum sampler with class-sensitive optimization to calibrate minority-class decision boundaries.</em></p>

DCIM treats imbalance as **both a representation problem and a classifier-calibration problem**.

**Phase 1 — representation-oriented learning.** From epoch 1 to \(T_1\), mini-batches are constructed with approximately balanced class exposure. Minority classes are oversampled toward a target count determined by the majority-class count and an oversampling intensity coefficient. This prevents majority gradients from dominating the embedding space.

**Phase 2 — calibration-oriented refinement.** From \(T_1+1\) to the final epoch \(T\), the sampling distribution is gradually annealed from the balanced target back toward each class's original frequency. This progressively restores realistic priors and corrects the boundary bias that pure oversampling can introduce.

DCIM further introduces a class-frequency-dependent margin

$$
\Delta_y = \frac{m_{max}}{n_y^{1/4}},
$$

so rarer classes receive a larger classification margin. The margin-adjusted probability is combined with **focal weighting**, concentrating optimization on hard examples while preserving stronger separation for tail classes.

The full objective combines classification loss with HMoEF's load-balancing and diversity regularizers.

---

# Experimental Results

The paper evaluates KAMEL on **25 public tabular classification datasets**: **13 binary** and **12 multiclass** tasks, against **12 representative baselines** spanning tree ensembles, deep tabular models, and tabular foundation models.

<table>
<tr>
<td align="center"><strong>25 / 25</strong><br/>best or tied-best F1</td>
<td align="center"><strong>+2.79%</strong><br/>avg. relative gain, binary</td>
<td align="center"><strong>+1.36%</strong><br/>avg. relative gain, multiclass</td>
<td align="center"><strong>5</strong><br/>datasets at 100% F1</td>
</tr>
</table>

### Main findings

- KAMEL obtains the **best or tied-best F1-score on all 25 datasets**.
- Average relative improvement over the strongest competing baseline is **2.79%** on binary tasks and **1.36%** on multiclass tasks.
- Relative gains reach **4.19% on Heart Disease** and **5.13% on Solar Flare**.
- KAMEL reaches **100.0% F1** on Banknote, MONK's-1, Car Evaluation, Cardiotocography, and Nursery; it is the only compared method to reach 100.0% F1 on Nursery.

### What the ablations show

| Ablation / replacement | Reported average relative F1 change |
|---|---:|
| UKAE/RKAN → standard MLP | **−11.62%** |
| Remove tabular modality | **−10.58%** |
| Remove visual modality | **−6.84%** |
| Remove textual modality | **−9.81%** |
| HMoEF → concatenation | **−7.10%** |
| HMoEF → late fusion | **−6.23%** |
| DCIM → single-stage cross-entropy | **−7.59%** |
| DCIM → single-stage focal loss | **−7.82%** |

These results support the paper's central claim that performance comes from the **coordination** of multimodal representation, RKAN encoding, heterogeneous routing, and curriculum imbalance mitigation—not from a single isolated module.

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
├── assets/paper/                  # Figures extracted from the published paper
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

Baseline models may require additional model-specific dependencies such as XGBoost, LightGBM, CatBoost, or packages required by individual deep tabular baselines.

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
> The command above documents the **current repository interface**. For the exact experimental settings reported in the paper, consult **Section 5.1 and Appendix D** of the publication, especially the dataset-specific hyperparameters selected by Bayesian optimization.

### Key CLI arguments

| Argument | Choices / type | Current default | Description |
|---|---|---:|---|
| `--dataset` | supported dataset alias | required | Dataset configuration. |
| `--data_path` | path | `None` | Explicit local dataset path. |
| `--model_size` | `small`, `base`, `large`, `xlarge` | `base` | Model-scale preset. |
| `--kan_type` | `kan`, `fast_kan`, `cheby_kan`, `wave_kan` | `fast_kan` | Current code's KAN implementation switch; see the paper/code naming note above. |
| `--epochs` | int | `10` | Total training epochs. |
| `--phase1_epochs` | int | `None` | Override the first curriculum phase length. |
| `--sampling_strategy` | `curriculum`, `balanced`, `smote` | `curriculum` | Sampling strategy. |
| `--loss_strategy` | `ldam`, `focal`, `adaptive`, `cross_entropy` | `ldam` | Current code loss switch. |
| `--device` | `cuda`, `cpu` | `cuda` | Training device. |
| `--seed` | int | `42` | Random seed. |

If CUDA is requested but unavailable, the training entry point falls back to CPU.

### Dataset files

The repository contains dataset configuration metadata under `kamel/data/`, but the current root tree does **not** vendor a complete `data/` directory with all benchmark CSV files. Supply your local dataset path explicitly when needed:

```bash
python train_kamel.py \
  --dataset spambase \
  --data_path /path/to/spambase.data \
  --device cuda
```

### Run baseline comparisons

```bash
# All configured baselines for one dataset
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

The article was accepted on **19 July 2026** and became available online on **20 July 2026**.

# Citation

If KAMEL is useful in your research, please cite the published article:

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

**SSMT · UKAE/RKAN · HMoEF · DCIM**

<sub>Learning tabular representations that remain expressive under heterogeneity and class imbalance.</sub>

</div>
