"""
GANDALF (Gated Adaptive Network for Deep Automated Learning of Features) implementation.

Based on "GANDALF: Gated Adaptive Network for Deep Automated Learning of Features"
(Joseph & Raj, 2022)
Paper: https://arxiv.org/abs/2207.08548
Reference: https://github.com/manujosephv/pytorch_tabular

This is a faithful reproduction of the GFLU (Gated Feature Learning Unit) architecture
with t-softmax for sparse feature masks and Beta-distribution initialization.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from typing import Dict, Any, Tuple, Optional
import warnings
import math

warnings.filterwarnings('ignore')


# ============================================================================
# T-Softmax — weighted sparse softmax (Eq. 5 in the paper)
# ============================================================================

class TSoftmax(nn.Module):
    """
    T-softmax: a learnable temperature softmax that produces sparse outputs.

    For each feature i:
        t-softmax(z_i) = max(0,  exp(z_i / T) / sum_j exp(z_j / T)  -  1/d )

    where d = number of features and T is a learnable temperature.
    This produces sparse attention masks while remaining differentiable.
    """

    def __init__(self, dim: int, learnable: bool = True):
        super().__init__()
        self.dim = dim
        if learnable:
            self.t = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('t', torch.ones(1))

    def forward(self, x):
        # x: (..., dim)
        t = self.t.clamp(min=0.01)
        scores = torch.softmax(x / t, dim=-1)
        threshold = 1.0 / self.dim
        sparse_scores = F.relu(scores - threshold)
        # Re-normalise so they sum to 1 (or close to it)
        denom = sparse_scores.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return sparse_scores / denom


# ============================================================================
# GFLU — Gated Feature Learning Unit (Sec. 3.2 of the paper)
# ============================================================================

class GFLU(nn.Module):
    """
    Gated Feature Learning Unit.

    Each GFLU stage performs:
        1) Feature mask via t-softmax(W_mask @ h + b_mask)
        2) Gated transformation:
              r = W2 @ (mask * h) + b2
              z = sigmoid(W_gate @ (mask * h) + b_gate)
              h_next = z * r + (1 - z) * h

    Parameters
    ----------
    n_features : int
        Number of input features.
    n_stages : int
        Number of stacked GFLU stages.
    dropout : float
        Dropout rate between stages.
    learnable_sparsity : bool
        Whether the t-softmax temperature is learnable.
    feature_init_sparsity : float
        Concentration parameter for Beta-distribution initialisation of masks.
    """

    def __init__(self, n_features: int, n_stages: int = 6, dropout: float = 0.0,
                 learnable_sparsity: bool = True, feature_init_sparsity: float = 0.3):
        super().__init__()
        self.n_features = n_features
        self.n_stages = n_stages

        # Per-stage parameters
        self.mask_linears = nn.ModuleList()
        self.t_softmax_layers = nn.ModuleList()
        self.W2 = nn.ModuleList()
        self.gate_linears = nn.ModuleList()
        self.stage_norms = nn.ModuleList()

        for _ in range(n_stages):
            # Feature mask
            mask_linear = nn.Linear(n_features, n_features)
            self.mask_linears.append(mask_linear)
            self.t_softmax_layers.append(TSoftmax(n_features, learnable=learnable_sparsity))

            # Gated transform
            self.W2.append(nn.Linear(n_features, n_features))
            self.gate_linears.append(nn.Linear(n_features, n_features))
            self.stage_norms.append(nn.BatchNorm1d(n_features))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Beta-distribution initialisation for mask diversity
        self._init_masks(feature_init_sparsity)

    def _init_masks(self, sparsity: float):
        """Initialise mask linear weights from Beta distribution for diversity."""
        alpha = 1.0 - sparsity
        beta_param = sparsity
        alpha = max(alpha, 0.05)
        beta_param = max(beta_param, 0.05)
        for i, ml in enumerate(self.mask_linears):
            with torch.no_grad():
                # Sample from Beta and shift to [-1, 1]
                beta_samples = torch.distributions.Beta(alpha, beta_param).sample(ml.weight.shape)
                ml.weight.copy_(2.0 * beta_samples - 1.0)
                nn.init.zeros_(ml.bias)

    def forward(self, x):
        # x: (batch, n_features)
        h = x
        for i in range(self.n_stages):
            # Feature mask
            mask_logits = self.mask_linears[i](h)
            mask = self.t_softmax_layers[i](mask_logits)  # sparse mask in [0, 1]

            # Gated transformation
            masked_h = mask * h
            r = self.W2[i](masked_h)
            z = torch.sigmoid(self.gate_linears[i](masked_h))
            h = z * r + (1 - z) * h
            h = self.stage_norms[i](h)
            h = self.dropout(h)
        return h


# ============================================================================
# Full GANDALF model
# ============================================================================

class GANDALFModel(nn.Module):
    """
    GANDALF: GFLU backbone + MLP classification head.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    n_classes : int
        Number of target classes.
    gflu_stages : int
        Number of GFLU stages.
    gflu_dropout : float
        Dropout within GFLU stages.
    learnable_sparsity : bool
        Whether t-softmax temperature is learnable.
    feature_init_sparsity : float
        Beta distribution sparsity param.
    head_hidden : int
        Hidden units in the MLP head.
    head_dropout : float
        Dropout in the MLP head.
    """

    def __init__(self, input_dim, n_classes, gflu_stages=6, gflu_dropout=0.0,
                 learnable_sparsity=True, feature_init_sparsity=0.3,
                 head_hidden=128, head_dropout=0.1):
        super().__init__()
        self.gflu = GFLU(
            n_features=input_dim,
            n_stages=gflu_stages,
            dropout=gflu_dropout,
            learnable_sparsity=learnable_sparsity,
            feature_init_sparsity=feature_init_sparsity
        )
        self.head = nn.Sequential(
            nn.Linear(input_dim, head_hidden),
            nn.BatchNorm1d(head_hidden),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.BatchNorm1d(head_hidden // 2),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden // 2, n_classes)
        )

    def forward(self, x):
        h = self.gflu(x)
        return self.head(h)


# ============================================================================
# Wrapper matching project API: __init__, train, predict, get_model_info
# ============================================================================

class GANDALFClassifier_ML:
    """
    GANDALF classifier wrapper for tabular classification.

    Faithfully reproduces the GFLU architecture with t-softmax feature masks
    and Beta-distribution initialisation from the original paper.
    """

    def __init__(self, **kwargs):
        # Architecture
        self.gflu_stages = kwargs.get('gflu_stages', 6)
        self.gflu_dropout = kwargs.get('gflu_dropout', 0.0)
        self.learnable_sparsity = kwargs.get('learnable_sparsity', True)
        self.feature_init_sparsity = kwargs.get('feature_init_sparsity', 0.3)
        self.head_hidden = kwargs.get('head_hidden', 128)
        self.head_dropout = kwargs.get('head_dropout', 0.1)

        # Training
        self.learning_rate = kwargs.get('learning_rate', 1e-3)
        self.batch_size = kwargs.get('batch_size', 256)
        self.max_epochs = kwargs.get('max_epochs', 50)
        self.patience = kwargs.get('patience', 8)
        self.weight_decay = kwargs.get('weight_decay', 1e-5)
        self.random_state = kwargs.get('random_state', 42)

        # Preprocessing & state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')
        self.cat_encoder = None
        self.cat_cols_ = None
        self.is_fitted = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = kwargs

        # Reproducibility
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

    # ------------------------------------------------------------------ train
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:

        # Preprocess — handle categorical columns
        X_tr = self._encode_categoricals_fit(X_train)
        X_tr = self.imputer.fit_transform(X_tr)
        X_tr = self.scaler.fit_transform(X_tr)
        y_tr = self.label_encoder.fit_transform(y_train)
        X_v = self._encode_categoricals_transform(X_val)
        X_v = self.scaler.transform(self.imputer.transform(X_v))
        y_v = self.label_encoder.transform(y_val)

        n_features = X_tr.shape[1]
        n_classes = len(np.unique(y_tr))

        # Build model
        self.model = GANDALFModel(
            input_dim=n_features,
            n_classes=n_classes,
            gflu_stages=self.gflu_stages,
            gflu_dropout=self.gflu_dropout,
            learnable_sparsity=self.learnable_sparsity,
            feature_init_sparsity=self.feature_init_sparsity,
            head_hidden=self.head_hidden,
            head_dropout=self.head_dropout
        ).to(self.device)

        # Tensors
        X_tr_t = torch.FloatTensor(X_tr).to(self.device)
        y_tr_t = torch.LongTensor(y_tr).to(self.device)
        X_v_t = torch.FloatTensor(X_v).to(self.device)
        y_v_t = torch.LongTensor(y_v).to(self.device)

        # Class-weighted loss
        class_counts = np.bincount(y_tr, minlength=n_classes).astype(float)
        class_weights = 1.0 / np.maximum(class_counts, 1)
        class_weights /= class_weights.sum()
        class_weights *= n_classes
        criterion = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(class_weights).to(self.device)
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=max(1, self.patience // 3)
        )

        best_val_loss = float('inf')
        best_state = None
        patience_ctr = 0

        for epoch in range(self.max_epochs):
            self.model.train()
            indices = torch.randperm(len(X_tr_t))
            for i in range(0, len(X_tr_t), self.batch_size):
                idx = indices[i:i + self.batch_size]
                optimizer.zero_grad()
                logits = self.model(X_tr_t[idx])
                loss = criterion(logits, y_tr_t[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_v_t)
                val_loss = criterion(val_logits, y_v_t).item()
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
            if patience_ctr >= self.patience:
                break

        # Restore best
        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        self.is_fitted = True

        # Validation probabilities
        self.model.eval()
        with torch.no_grad():
            val_proba = torch.softmax(self.model(X_v_t), dim=1).cpu().numpy()

        if n_classes == 2:
            val_proba = val_proba[:, 1]

        return {
            'val_proba': val_proba,
            'training_samples': len(X_tr),
            'n_epochs': epoch + 1,
            'best_val_loss': best_val_loss,
            'model_info': self.get_model_info()
        }

    # ---------------------------------------------------------------- predict
    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_t = self._encode_categoricals_transform(X_test)
        X_t = self.scaler.transform(self.imputer.transform(X_t))
        X_t = torch.FloatTensor(X_t).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        y_pred = self.label_encoder.inverse_transform(preds)
        n_classes = proba.shape[1]
        y_proba = proba[:, 1] if n_classes == 2 else proba
        return y_pred, y_proba

    # ----------------------------------------- categorical encoding helpers
    def _encode_categoricals_fit(self, X: np.ndarray) -> np.ndarray:
        """Detect and ordinal-encode non-numeric columns, fit encoders."""
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.cat_cols_ = []
        for j in range(X.shape[1]):
            try:
                X[:, j].astype(np.float64)
            except (ValueError, TypeError):
                self.cat_cols_.append(j)
        if not self.cat_cols_:
            return X.astype(np.float64)
        self.cat_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_out = X.copy()
        self.cat_encoder.fit(X_out[:, self.cat_cols_].astype(str))
        X_out[:, self.cat_cols_] = self.cat_encoder.transform(X_out[:, self.cat_cols_].astype(str))
        return X_out.astype(np.float64)

    def _encode_categoricals_transform(self, X: np.ndarray) -> np.ndarray:
        """Transform non-numeric columns using fitted encoders."""
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            X = X.values
        if not self.cat_cols_:
            return X.astype(np.float64)
        X_out = X.copy()
        X_out[:, self.cat_cols_] = self.cat_encoder.transform(X_out[:, self.cat_cols_].astype(str))
        return X_out.astype(np.float64)

    # --------------------------------------------------------- get_model_info
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'algorithm': 'GANDALF',
            'type': 'Gated Adaptive Network for Deep Automated Learning of Features',
            'description': 'Stacked Gated Feature Learning Units with t-softmax sparse feature masks',
            'paper': 'GANDALF: Gated Adaptive Network for Deep Automated Learning of Features (Joseph & Raj, 2022)',
            'arxiv': 'https://arxiv.org/abs/2207.08548',
            'preprocessing': 'SimpleImputer + StandardScaler + LabelEncoder',
            'parameters': {
                'gflu_stages': self.gflu_stages,
                'gflu_dropout': self.gflu_dropout,
                'learnable_sparsity': self.learnable_sparsity,
                'feature_init_sparsity': self.feature_init_sparsity,
                'head_hidden': self.head_hidden,
                'head_dropout': self.head_dropout,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'max_epochs': self.max_epochs,
                'patience': self.patience,
                'random_state': self.random_state
            },
            'model_state': {
                'is_fitted': self.is_fitted,
                'device': str(self.device)
            }
        }
