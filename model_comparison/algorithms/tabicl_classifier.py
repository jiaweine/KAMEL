import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import warnings
import math
import random
from typing import Optional, List, Tuple, Union, Dict, Any


class ColumnEmbedding(nn.Module):
    """Column-wise embedding layer that creates distribution-aware embeddings for each feature"""
    
    def __init__(self, n_features: int, embed_dim: int = 64):
        super().__init__()
        self.n_features = n_features
        self.embed_dim = embed_dim
        
        # Feature embedding layers
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(n_features)
        ])
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(n_features, embed_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, n_features)
        Returns:
            Embedded tensor of shape (batch_size, n_features, embed_dim)
        """
        batch_size = x.size(0)
        embedded_features = []
        
        for i in range(self.n_features):
            # Extract feature and add batch dimension for linear layer
            feature_val = x[:, i:i+1]  # (batch_size, 1)
            
            # Apply feature-specific embedding
            embedded = self.feature_embeddings[i](feature_val)  # (batch_size, embed_dim)
            embedded_features.append(embedded)
        
        # Stack to get (batch_size, n_features, embed_dim)
        embedded = torch.stack(embedded_features, dim=1)
        
        # Add position embeddings
        positions = torch.arange(self.n_features, device=x.device)
        pos_embeds = self.position_embeddings(positions)  # (n_features, embed_dim)
        embedded = embedded + pos_embeds.unsqueeze(0)  # Broadcasting
        
        # Apply layer norm and dropout
        embedded = self.layer_norm(embedded)
        embedded = self.dropout(embedded)
        
        return embedded


class RowInteraction(nn.Module):
    """Row-wise interaction layer that captures interactions between features within each row"""
    
    def __init__(self, embed_dim: int = 64, n_heads: int = 8, n_layers: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, n_heads, dropout=0.1, batch_first=True)
            for _ in range(n_layers)
        ])
        
        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(0.1)
            )
            for _ in range(n_layers)
        ])
        
        # Layer normalizations
        self.layer_norms1 = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(n_layers)
        ])
        self.layer_norms2 = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(n_layers)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, n_features, embed_dim)
        Returns:
            Output tensor of shape (batch_size, n_features, embed_dim)
        """
        for i in range(self.n_layers):
            # Multi-head attention
            attn_out, _ = self.attention_layers[i](x, x, x)
            x = self.layer_norms1[i](x + attn_out)
            
            # Feed-forward network
            ffn_out = self.ffn_layers[i](x)
            x = self.layer_norms2[i](x + ffn_out)
        
        return x


class InContextLearning(nn.Module):
    """Dataset-wise in-context learning layer"""
    
    def __init__(self, embed_dim: int = 64, n_heads: int = 8, n_layers: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Cross-attention layers for learning from labeled examples
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, n_heads, dropout=0.1, batch_first=True)
            for _ in range(n_layers)
        ])
        
        # Self-attention layers
        self.self_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, n_heads, dropout=0.1, batch_first=True)
            for _ in range(n_layers)
        ])
        
        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(0.1)
            )
            for _ in range(n_layers)
        ])
        
        # Layer normalizations
        self.layer_norms1 = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(n_layers)
        ])
        self.layer_norms2 = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(n_layers)
        ])
        self.layer_norms3 = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(n_layers)
        ])
        
    def forward(self, query_features: torch.Tensor, support_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_features: Query features of shape (batch_size, n_features, embed_dim)
            support_features: Support features of shape (support_size, n_features, embed_dim)
        Returns:
            Output tensor of shape (batch_size, n_features, embed_dim)
        """
        # Flatten features for sequence modeling
        batch_size, n_features, embed_dim = query_features.shape
        query_seq = query_features.view(batch_size, n_features, embed_dim)
        support_seq = support_features.view(support_features.size(0), n_features, embed_dim)
        
        x = query_seq
        
        for i in range(self.n_layers):
            # Cross-attention with support set
            if support_seq.size(0) > 0:
                cross_attn_out, _ = self.cross_attention_layers[i](x, support_seq, support_seq)
                x = self.layer_norms1[i](x + cross_attn_out)
            
            # Self-attention
            self_attn_out, _ = self.self_attention_layers[i](x, x, x)
            x = self.layer_norms2[i](x + self_attn_out)
            
            # Feed-forward network
            ffn_out = self.ffn_layers[i](x)
            x = self.layer_norms3[i](x + ffn_out)
        
        return x


class TabICLModel(nn.Module):
    """Complete TabICL model architecture"""
    
    def __init__(self, n_features: int, n_classes: int, embed_dim: int = 64, 
                 n_heads: int = 8, row_layers: int = 2, icl_layers: int = 4):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.embed_dim = embed_dim
        
        # Three main components
        self.column_embedding = ColumnEmbedding(n_features, embed_dim)
        self.row_interaction = RowInteraction(embed_dim, n_heads, row_layers)
        self.in_context_learning = InContextLearning(embed_dim, n_heads, icl_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * n_features, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, n_classes)
        )
        
        # Special token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
    def forward(self, query_x: torch.Tensor, support_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query_x: Query samples of shape (batch_size, n_features)
            support_x: Support samples of shape (support_size, n_features)
        Returns:
            Logits of shape (batch_size, n_classes)
        """
        # Column-wise embedding
        query_embedded = self.column_embedding(query_x)  # (batch_size, n_features, embed_dim)
        
        # Row-wise interaction
        query_interacted = self.row_interaction(query_embedded)
        
        # Prepare support set if available
        support_interacted = None
        if support_x is not None and support_x.size(0) > 0:
            support_embedded = self.column_embedding(support_x)
            support_interacted = self.row_interaction(support_embedded)
        
        # In-context learning
        if support_interacted is not None:
            query_contextualized = self.in_context_learning(query_interacted, support_interacted)
        else:
            query_contextualized = query_interacted
        
        # Global pooling and classification
        batch_size = query_contextualized.size(0)
        pooled = query_contextualized.view(batch_size, -1)  # Flatten
        logits = self.classifier(pooled)
        
        return logits


class TabICLClassifier(BaseEstimator, ClassifierMixin):
    """TabICL Classifier implementation"""
    
    def __init__(self, 
                 n_estimators: int = 32,
                 embed_dim: int = 64,
                 n_heads: int = 8,
                 row_layers: int = 2,
                 icl_layers: int = 4,
                 learning_rate: float = 1e-3,
                 batch_size: int = 64,
                 max_epochs: int = 100,
                 patience: int = 10,
                 outlier_threshold: float = 4.0,
                 softmax_temperature: float = 0.9,
                 class_shift: bool = True,
                 feat_shuffle_method: str = "latin",
                 norm_methods: List[str] = None,
                 device: Optional[str] = None,
                 random_state: int = 42,
                 verbose: bool = False):
        
        self.n_estimators = n_estimators
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.row_layers = row_layers
        self.icl_layers = icl_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.outlier_threshold = outlier_threshold
        self.softmax_temperature = softmax_temperature
        self.class_shift = class_shift
        self.feat_shuffle_method = feat_shuffle_method
        self.norm_methods = norm_methods or ["none", "power"]
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        
        # Initialize components
        self.models = []
        self.scalers = []
        self.label_encoder = None
        self.n_features_ = None
        self.n_classes_ = None
        self.classes_ = None
        
        # Set random seeds
        self._set_random_seeds()
        
    def _set_random_seeds(self):
        """Set random seeds for reproducibility"""
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)
    
    def _get_device(self):
        """Get the appropriate device"""
        if self.device is not None:
            return torch.device(self.device)
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _preprocess_features(self, X: np.ndarray, fit: bool = False) -> List[np.ndarray]:
        """Preprocess features with different normalization methods"""
        X_variants = []
        
        for norm_method in self.norm_methods:
            if fit:
                if norm_method == "none":
                    scaler = None
                    X_norm = X.copy()
                elif norm_method == "standard":
                    scaler = StandardScaler()
                    X_norm = scaler.fit_transform(X)
                elif norm_method == "power":
                    # Power transformation (simple version)
                    scaler = StandardScaler()
                    X_pow = np.sign(X) * np.power(np.abs(X) + 1e-8, 0.5)
                    X_norm = scaler.fit_transform(X_pow)
                else:
                    scaler = StandardScaler()
                    X_norm = scaler.fit_transform(X)
                
                self.scalers.append(scaler)
            else:
                scaler = self.scalers[len(X_variants)]
                if scaler is None:
                    X_norm = X.copy()
                elif norm_method == "power":
                    X_pow = np.sign(X) * np.power(np.abs(X) + 1e-8, 0.5)
                    X_norm = scaler.transform(X_pow)
                else:
                    X_norm = scaler.transform(X)
            
            # Outlier clipping
            if self.outlier_threshold > 0:
                std_vals = np.std(X_norm, axis=0)
                mean_vals = np.mean(X_norm, axis=0)
                X_norm = np.clip(X_norm, 
                               mean_vals - self.outlier_threshold * std_vals,
                               mean_vals + self.outlier_threshold * std_vals)
            
            X_variants.append(X_norm)
        
        return X_variants
    
    def _create_feature_permutations(self, n_features: int) -> List[List[int]]:
        """Create feature permutations for ensemble diversity"""
        permutations = []
        
        if self.feat_shuffle_method == "latin":
            # Latin square-like permutations
            for i in range(self.n_estimators):
                perm = list(range(n_features))
                # Apply different rotation based on estimator index
                rotation = (i * n_features // self.n_estimators) % n_features
                perm = perm[rotation:] + perm[:rotation]
                permutations.append(perm)
        else:
            # Random permutations
            for i in range(self.n_estimators):
                perm = list(range(n_features))
                np.random.shuffle(perm)
                permutations.append(perm)
        
        return permutations
    
    def _apply_class_shift(self, y: np.ndarray, shift: int) -> np.ndarray:
        """Apply cyclic class shift"""
        if not self.class_shift or shift == 0:
            return y
        return (y + shift) % self.n_classes_
    
    def fit(self, X, y):
        """Fit the TabICL model"""
        if self.verbose:
            print("Fitting TabICL model...")
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.classes_ = self.label_encoder.classes_
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        
        # Preprocess features
        X_variants = self._preprocess_features(X, fit=True)
        
        # Create feature permutations
        feature_perms = self._create_feature_permutations(self.n_features_)
        
        # Get device
        device = self._get_device()
        
        # Train ensemble of models
        self.models = []
        
        for i in range(self.n_estimators):
            if self.verbose:
                print(f"Training model {i+1}/{self.n_estimators}")
            
            # Select data variant and permutation
            X_var = X_variants[i % len(X_variants)]
            perm = feature_perms[i]
            X_perm = X_var[:, perm]
            
            # Apply class shift
            shift = i % self.n_classes_ if self.class_shift else 0
            y_shift = self._apply_class_shift(y_encoded, shift)
            
            # Create model
            model = TabICLModel(
                n_features=self.n_features_,
                n_classes=self.n_classes_,
                embed_dim=self.embed_dim,
                n_heads=self.n_heads,
                row_layers=self.row_layers,
                icl_layers=self.icl_layers
            ).to(device)
            
            # Train model
            self._train_single_model(model, X_perm, y_shift, device)
            
            # Store model info
            self.models.append({
                'model': model,
                'permutation': perm,
                'class_shift': shift,
                'variant_idx': i % len(X_variants)
            })
        
        return self
    
    def _train_single_model(self, model, X, y, device):
        """Train a single model"""
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.LongTensor(y).to(device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Optimizer and loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Forward pass (no support set during training)
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        
        model.eval()
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if not self.models:
            raise ValueError("Model not fitted yet")
        
        X = np.array(X)
        device = self._get_device()
        
        # Preprocess features
        X_variants = self._preprocess_features(X, fit=False)
        
        all_probas = []
        
        for i, model_info in enumerate(self.models):
            model = model_info['model']
            perm = model_info['permutation']
            shift = model_info['class_shift']
            variant_idx = model_info['variant_idx']
            
            # Apply same preprocessing as training
            X_var = X_variants[variant_idx]
            X_perm = X_var[:, perm]
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(X_perm).to(device)
            
            # Predict
            model.eval()
            with torch.no_grad():
                logits = model(X_tensor)
                # Apply temperature scaling
                logits = logits / self.softmax_temperature
                probas = F.softmax(logits, dim=1)
            
            # Reverse class shift
            if self.class_shift and shift > 0:
                # Reorder probabilities to undo the shift
                probas_unshift = torch.zeros_like(probas)
                for j in range(self.n_classes_):
                    original_class = (j - shift) % self.n_classes_
                    probas_unshift[:, original_class] = probas[:, j]
                probas = probas_unshift
            
            all_probas.append(probas.cpu().numpy())
        
        # Ensemble averaging
        avg_probas = np.mean(all_probas, axis=0)
        
        return avg_probas
    
    def predict(self, X):
        """Predict class labels"""
        probas = self.predict_proba(X)
        predictions = np.argmax(probas, axis=1)
        return self.label_encoder.inverse_transform(predictions)


class TabICLClassifier_ML:
    """TabICL Classifier wrapper for ML comparison framework"""
    
    def __init__(self, **kwargs):
        self.model = TabICLClassifier(**kwargs)
        self.kwargs = kwargs
        self.is_fitted = False
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the TabICL model"""
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        val_proba = None
        if X_val is not None:
            try:
                val_proba = self.model.predict_proba(X_val)
            except Exception:
                val_proba = None
        return {
            'val_proba': val_proba,
            'training_samples': len(X_train)
        }
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)
        return y_pred, y_proba
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)

    def get_model_info(self):
        device = str(self.model._get_device())
        return {
            'name': 'TabICL (reimplemented)',
            'architecture': {
                'embed_dim': self.model.embed_dim,
                'n_heads': self.model.n_heads,
                'row_layers': self.model.row_layers,
                'icl_layers': self.model.icl_layers,
                'n_estimators': self.model.n_estimators
            },
            'preprocessing': {
                'norm_methods': self.model.norm_methods,
                'outlier_threshold': self.model.outlier_threshold,
                'class_shift': self.model.class_shift,
                'feat_shuffle_method': self.model.feat_shuffle_method
            },
            'device': device
        }


def train_algorithm(X_train, y_train, X_val, y_val, algorithm_config, device='cuda'):
    """Train TabICL algorithm"""
    
    # Get configuration
    config = algorithm_config.get('tabicl', {})
    
    # Initialize model
    model = TabICLClassifier(
        n_estimators=config.get('n_estimators', 32),
        embed_dim=config.get('embed_dim', 64),
        n_heads=config.get('n_heads', 8),
        row_layers=config.get('row_layers', 2),
        icl_layers=config.get('icl_layers', 4),
        learning_rate=config.get('learning_rate', 1e-3),
        batch_size=config.get('batch_size', 64),
        max_epochs=config.get('max_epochs', 100),
        patience=config.get('patience', 10),
        outlier_threshold=config.get('outlier_threshold', 4.0),
        softmax_temperature=config.get('softmax_temperature', 0.9),
        class_shift=config.get('class_shift', True),
        feat_shuffle_method=config.get('feat_shuffle_method', 'latin'),
        norm_methods=config.get('norm_methods', ['none', 'power']),
        device=device,
        random_state=config.get('random_state', 42),
        verbose=config.get('verbose', False)
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    return model


def evaluate_algorithm(model, X_test, y_test):
    """Evaluate TabICL algorithm"""
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # AUC-ROC for binary/multiclass
    try:
        if len(np.unique(y_test)) == 2:
            auc_roc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            auc_roc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    except:
        auc_roc = 0.0
    
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'auc_roc': auc_roc,
        'auc_pr': 0.0  # Placeholder
    }
    
    return metrics, y_pred, y_pred_proba
