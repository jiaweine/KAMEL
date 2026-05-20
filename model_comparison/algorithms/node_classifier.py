"""
NODE (Neural Oblivious Decision Ensembles) implementation.

Based on "Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data"
(Popov, Morozov, Babenko, 2019)
Paper: https://arxiv.org/abs/1909.06312
Official Implementation: https://github.com/Qwicen/node

This is a faithful reproduction of the ODST (Oblivious Differentiable Sparsemax Trees)
and DenseBlock architecture from the original repository.
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
# Sparsemax (Martins & Astudillo, 2016) — exact PyTorch implementation
# ============================================================================

class SparsemaxFunction(torch.autograd.Function):
    """Sparsemax activation: projects onto the probability simplex."""

    @staticmethod
    def forward(ctx, input):
        dim = -1
        sorted_input, _ = torch.sort(input, descending=True, dim=dim)
        cumsum = torch.cumsum(sorted_input, dim=dim)
        k = torch.arange(1, input.size(dim) + 1, device=input.device, dtype=input.dtype)
        for _ in range(input.dim() - 1):
            k = k.unsqueeze(0)
        support = (sorted_input - (cumsum - 1) / k) > 0
        k_max = support.sum(dim=dim, keepdim=True).clamp(min=1)
        tau = (cumsum.gather(dim, (k_max - 1).long()) - 1) / k_max.float()
        output = torch.clamp(input - tau, min=0)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        nonzeros = (output > 0).float()
        s = nonzeros.sum(dim=-1, keepdim=True).clamp(min=1)
        grad_input = nonzeros * (grad_output - (grad_output * nonzeros).sum(dim=-1, keepdim=True) / s)
        return grad_input


def sparsemax(input):
    return SparsemaxFunction.apply(input)


def sparsemoid(input):
    """Sparsemoid: a sparse version of sigmoid in [0, 1]."""
    return (0.5 * input + 0.5).clamp(0, 1)


# ============================================================================
# ODST — Oblivious Differentiable Sparsemax Tree
# ============================================================================

class ODST(nn.Module):
    """
    Oblivious Differentiable Sparsemax Tree (ODST).

    Each tree selects features via sparsemax, computes soft comparisons
    with learned thresholds, and produces 2^depth response values that
    are mixed by the soft routing probabilities.

    Parameters
    ----------
    in_features : int
        Number of input features.
    num_trees : int
        Number of parallel trees in this layer.
    depth : int
        Depth of each oblivious decision tree.
    tree_dim : int
        Output dimension per tree (allows multi-dim tree responses).
    flatten_output : bool
        If True, output shape is (batch, num_trees * tree_dim).
    """

    def __init__(self, in_features, num_trees=1024, depth=6, tree_dim=1,
                 flatten_output=True):
        super().__init__()
        self.in_features = in_features
        self.num_trees = num_trees
        self.depth = depth
        self.tree_dim = tree_dim
        self.flatten_output = flatten_output

        # Feature selection weights: (in_features, num_trees, depth)
        self.feature_selection_logits = nn.Parameter(
            torch.zeros(in_features, num_trees, depth), requires_grad=True
        )
        # Thresholds for comparisons: (num_trees, depth)
        self.feature_thresholds = nn.Parameter(
            torch.zeros(num_trees, depth), requires_grad=True
        )
        # Log-temperatures for sigmoid sharpness: (num_trees, depth)
        self.log_temperatures = nn.Parameter(
            torch.zeros(num_trees, depth), requires_grad=True
        )
        # Response values at leaves: (num_trees, tree_dim, 2^depth)
        self.response = nn.Parameter(
            torch.zeros(num_trees, tree_dim, 2 ** depth), requires_grad=True
        )

        # Initialize
        nn.init.uniform_(self.feature_selection_logits, -1, 1)
        nn.init.uniform_(self.feature_thresholds, -1, 1)
        nn.init.zeros_(self.log_temperatures)
        nn.init.uniform_(self.response, -1, 1)

    def forward(self, input):
        # input: (batch, in_features)
        batch_size = input.shape[0]

        # Feature selection via sparsemax: (in_features, num_trees, depth)
        feature_selectors = sparsemax(self.feature_selection_logits)

        # Selected features: (batch, num_trees, depth)
        # = einsum('bf, ftd -> btd', input, feature_selectors)
        feature_values = torch.einsum('bi,ind->bnd', input, feature_selectors)

        # Soft threshold comparisons using sparsemoid
        # threshold_logits: (batch, num_trees, depth)
        temperatures = self.log_temperatures.exp().clamp(min=1e-6)
        threshold_logits = (feature_values - self.feature_thresholds) / temperatures
        threshold_logits = threshold_logits.clamp(-10, 10)

        # Binary routing probabilities: (batch, num_trees, depth) in [0, 1]
        bins = sparsemoid(threshold_logits)

        # Compute leaf probabilities via outer products over depth
        # Each leaf is indexed by a binary vector of length `depth`
        # bin_matches shape: (batch, num_trees, depth, 2)
        bin_matches = torch.stack([1 - bins, bins], dim=-1)  # (B, T, D, 2)

        # Compute product over depth -> (batch, num_trees, 2^depth)
        # Use iterative binary expansion
        response_weights = bin_matches[:, :, 0, :]  # (B, T, 2)
        for d in range(1, self.depth):
            # (B, T, 2^d) x (B, T, 2) -> (B, T, 2^(d+1))
            response_weights = (
                response_weights.unsqueeze(-1) * bin_matches[:, :, d, :].unsqueeze(-2)
            ).flatten(-2, -1)

        # response_weights: (B, num_trees, 2^depth)
        # self.response: (num_trees, tree_dim, 2^depth)
        # output: (B, num_trees, tree_dim)
        response = torch.einsum('btl,tdl->btd', response_weights, self.response)

        if self.flatten_output:
            return response.flatten(1, 2)  # (B, num_trees * tree_dim)
        return response


# ============================================================================
# DenseBlock — stacks ODST layers with dense (residual) connections
# ============================================================================

class DenseBlock(nn.Module):
    """
    Stacks multiple ODST layers with dense connections.

    Each subsequent layer receives as input the original features concatenated
    with the outputs of all previous layers (DenseNet-style).

    Parameters
    ----------
    input_dim : int
        Number of original input features.
    num_trees : int
        Number of trees per ODST layer.
    depth : int
        Tree depth per ODST layer.
    num_layers : int
        Number of ODST layers.
    tree_dim : int
        Output dimension per tree.
    input_dropout : float
        Dropout on the input to each layer.
    """

    def __init__(self, input_dim, num_trees=1024, depth=6, num_layers=1,
                 tree_dim=1, input_dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_trees = num_trees
        self.depth = depth
        self.num_layers = num_layers
        self.tree_dim = tree_dim
        self.input_dropout = input_dropout

        layers = []
        current_in = input_dim
        for i in range(num_layers):
            layer = ODST(
                in_features=current_in,
                num_trees=num_trees,
                depth=depth,
                tree_dim=tree_dim,
                flatten_output=True
            )
            layers.append(layer)
            current_in += num_trees * tree_dim

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()

    def forward(self, x):
        # x: (batch, input_dim)
        h = x
        for layer in self.layers:
            layer_in = self.dropout(h)
            layer_out = layer(layer_in)
            h = torch.cat([h, layer_out], dim=1)
        # Return only the concatenated ODST outputs (exclude original input)
        return h[:, self.input_dim:]


# ============================================================================
# Full NODE model
# ============================================================================

class NODEModel(nn.Module):
    """
    Complete NODE model: DenseBlock backbone + linear classification head.

    Parameters
    ----------
    input_dim : int
        Number of input features (after preprocessing).
    n_classes : int
        Number of target classes.
    num_trees : int
        Trees per ODST layer.
    depth : int
        Tree depth.
    num_layers : int
        Number of ODST layers (DenseBlock depth).
    tree_dim : int
        Output dimension per tree.
    input_dropout : float
        Dropout on each layer's input.
    """

    def __init__(self, input_dim, n_classes, num_trees=1024, depth=6,
                 num_layers=1, tree_dim=1, input_dropout=0.0):
        super().__init__()
        self.dense_block = DenseBlock(
            input_dim=input_dim,
            num_trees=num_trees,
            depth=depth,
            num_layers=num_layers,
            tree_dim=tree_dim,
            input_dropout=input_dropout
        )
        total_tree_output = num_layers * num_trees * tree_dim
        self.head = nn.Linear(total_tree_output, n_classes)

    def forward(self, x):
        h = self.dense_block(x)
        return self.head(h)


# ============================================================================
# Wrapper matching project API: __init__, train, predict, get_model_info
# ============================================================================

class NODEClassifier_ML:
    """
    NODE (Neural Oblivious Decision Ensembles) wrapper for tabular classification.

    Faithfully reproduces the ODST + DenseBlock architecture from the official
    Qwicen/node repository with sparsemax feature selection and sparsemoid
    threshold comparisons.
    """

    def __init__(self, **kwargs):
        # Architecture
        self.num_trees = kwargs.get('num_trees', 1024)
        self.depth = kwargs.get('depth', 6)
        self.num_layers = kwargs.get('num_layers', 2)
        self.tree_dim = kwargs.get('tree_dim', 3)
        self.input_dropout = kwargs.get('input_dropout', 0.0)

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
        self.cat_encoder = None  # OrdinalEncoder for categorical cols
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
        self.model = NODEModel(
            input_dim=n_features,
            n_classes=n_classes,
            num_trees=self.num_trees,
            depth=self.depth,
            num_layers=self.num_layers,
            tree_dim=self.tree_dim,
            input_dropout=self.input_dropout
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
        # Detect object/str columns
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
            'algorithm': 'NODE',
            'type': 'Neural Oblivious Decision Ensembles',
            'description': 'Deep ensembles of differentiable oblivious decision trees with sparsemax feature selection',
            'paper': 'Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data (Popov et al., 2019)',
            'github': 'https://github.com/Qwicen/node',
            'preprocessing': 'SimpleImputer + StandardScaler + LabelEncoder',
            'parameters': {
                'num_trees': self.num_trees,
                'depth': self.depth,
                'num_layers': self.num_layers,
                'tree_dim': self.tree_dim,
                'input_dropout': self.input_dropout,
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
