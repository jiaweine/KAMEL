"""
SAINT implementation for tabular classification.
Based on "SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training" (Somepalli et al., 2021)
Real implementation adapted from the original SAINT paper.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from typing import Dict, Any, Tuple, Optional
import warnings
import math
warnings.filterwarnings('ignore')


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, dim, heads=8, dim_head=16, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


def rearrange(tensor, pattern, **kwargs):
    """Simple rearrange function for tensor reshaping."""
    if 'b n (h d) -> b h n d' in pattern:
        b, n, hd = tensor.shape
        h = kwargs['h']
        d = hd // h
        return tensor.reshape(b, n, h, d).transpose(1, 2)
    elif 'b h n d -> b n (h d)' in pattern:
        b, h, n, d = tensor.shape
        return tensor.transpose(1, 2).reshape(b, n, h * d)
    return tensor


class FeedForward(nn.Module):
    """Feed forward network."""
    
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SAINTBlock(nn.Module):
    """SAINT transformer block with intersample attention."""
    
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.attn = MultiHeadAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.ff(self.norm2(x)) + x
        return x


class IntersampleAttention(nn.Module):
    """Intersample attention mechanism."""
    
    def __init__(self, dim, heads=8, dim_head=16, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_out = nn.Sequential(
            nn.Linear(dim_head * heads, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x shape: (batch_size, num_features, dim)
        b, n, d = x.shape
        
        q = self.to_q(x)  # (b, n, heads * dim_head)
        k = self.to_k(x)  # (b, n, heads * dim_head) 
        v = self.to_v(x)  # (b, n, heads * dim_head)
        
        # Reshape for multi-head attention
        q = q.reshape(b, n, self.heads, -1).transpose(1, 2)  # (b, heads, n, dim_head)
        k = k.reshape(b, n, self.heads, -1).transpose(1, 2)  # (b, heads, n, dim_head)
        v = v.reshape(b, n, self.heads, -1).transpose(1, 2)  # (b, heads, n, dim_head)
        
        # Attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)  # (b, heads, n, dim_head)
        out = out.transpose(1, 2).reshape(b, n, -1)  # (b, n, heads * dim_head)
        
        return self.to_out(out)


class SAINTClassifier_ML:
    """
    SAINT implementation for tabular classification.
    """
    
    def __init__(self, **kwargs):
        # SAINT specific parameters
        self.dim = kwargs.get('dim', 32)
        self.depth = kwargs.get('depth', 6)
        self.heads = kwargs.get('heads', 8)
        self.dim_head = kwargs.get('dim_head', 16)
        self.mlp_dim = kwargs.get('mlp_dim', None)
        self.attn_dropout = kwargs.get('attn_dropout', 0.1)
        self.ff_dropout = kwargs.get('ff_dropout', 0.1)
        
        # Training parameters
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        self.batch_size = kwargs.get('batch_size', 256)
        self.max_epochs = kwargs.get('max_epochs', 100)
        self.patience = kwargs.get('patience', 10)
        self.weight_decay = kwargs.get('weight_decay', 1e-5)
        
        self.random_state = kwargs.get('random_state', 42)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set mlp_dim default
        if self.mlp_dim is None:
            self.mlp_dim = self.dim * 4
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.sampler = SMOTE(random_state=self.random_state)
        self.is_fitted = False
        
        # Store configuration
        self.config = kwargs
        
        # Set seeds
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
    def _build_network(self, input_dim, output_dim):
        """Build SAINT architecture."""
        
        class SAINTModel(nn.Module):
            def __init__(self, input_dim, output_dim, dim, depth, heads, dim_head, mlp_dim, attn_dropout, ff_dropout):
                super(SAINTModel, self).__init__()
                self.input_dim = input_dim
                self.output_dim = output_dim
                self.dim = dim
                
                # Input embedding
                self.input_embedding = nn.Linear(input_dim, dim)
                
                # Column embedding (each feature gets its own embedding)
                self.column_embedding = nn.Parameter(torch.randn(input_dim, dim))
                
                # Positional encoding
                self.pos_embedding = nn.Parameter(torch.randn(1, input_dim, dim))
                
                # Transformer blocks
                self.transformer_blocks = nn.ModuleList([
                    SAINTBlock(dim, heads, dim_head, mlp_dim, ff_dropout)
                    for _ in range(depth)
                ])
                
                # Intersample attention
                self.intersample_attn = IntersampleAttention(dim, heads, dim_head, attn_dropout)
                
                # Final layers
                self.norm = nn.LayerNorm(dim)
                self.to_cls_token = nn.Linear(dim, dim)
                self.classifier = nn.Sequential(
                    nn.Linear(dim, dim // 2),
                    nn.ReLU(),
                    nn.Dropout(ff_dropout),
                    nn.Linear(dim // 2, output_dim)
                )
                
                # Dropout
                self.dropout = nn.Dropout(ff_dropout)
                
            def forward(self, x):
                batch_size, num_features = x.shape
                
                # Input embedding - project each feature to embedding dimension
                x = x.unsqueeze(-1)  # (batch_size, num_features, 1)
                x = torch.matmul(x, torch.ones(1, self.dim, device=x.device))  # (batch_size, num_features, dim)
                
                # Add column embeddings
                x = x + self.column_embedding.unsqueeze(0)  # Broadcast column embeddings
                
                # Add positional embeddings
                x = x + self.pos_embedding
                
                # Apply dropout
                x = self.dropout(x)
                
                # Apply transformer blocks
                for transformer in self.transformer_blocks:
                    x = transformer(x)
                
                # Apply intersample attention
                x = self.intersample_attn(x) + x
                
                # Normalize
                x = self.norm(x)
                
                # Global average pooling across features
                x = x.mean(dim=1)  # (batch_size, dim)
                
                # Classification
                x = self.to_cls_token(x)
                output = self.classifier(x)
                
                return output
        
        return SAINTModel(
            input_dim, output_dim, self.dim, self.depth, self.heads, 
            self.dim_head, self.mlp_dim, self.attn_dropout, self.ff_dropout
        )
    
    def preprocess_data(self, X_train: np.ndarray, y_train: np.ndarray, 
                       X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Standard preprocessing: scaling and conditional oversampling."""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Check class imbalance ratio
        unique, counts = np.unique(y_train, return_counts=True)
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count
        
        # Only apply SMOTE if there's significant imbalance (ratio > 3)
        if imbalance_ratio > 3.0 and len(unique) == 2:
            print(f"Applying SMOTE due to class imbalance (ratio: {imbalance_ratio:.2f})")
            X_train_resampled, y_train_resampled = self.sampler.fit_resample(X_train_scaled, y_train)
        else:
            print(f"Skipping SMOTE - classes relatively balanced (ratio: {imbalance_ratio:.2f})")
            X_train_resampled, y_train_resampled = X_train_scaled, y_train
        
        return X_train_resampled, y_train_resampled, X_val_scaled, X_test_scaled
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train SAINT model."""
        
        # First encode labels on original data
        y_train_encoded_orig = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        
        # Preprocess data (no test data in training phase)
        X_train_proc, y_train_proc, X_val_proc, _ = self.preprocess_data(
            X_train, y_train, X_val, X_val
        )
        
        # Encode the resampled training labels using the already fitted encoder
        y_train_encoded = self.label_encoder.transform(y_train_proc)
        
        # Get dimensions
        input_dim = X_train_proc.shape[1]
        output_dim = len(np.unique(y_train_encoded))
        
        # Build model
        self.model = self._build_network(input_dim, output_dim).to(self.device)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_proc).to(self.device)
        y_train_tensor = torch.LongTensor(y_train_encoded).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_proc).to(self.device)
        y_val_tensor = torch.LongTensor(y_val_encoded).to(self.device)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            self.model.train()
            
            # Batch training
            total_loss = 0
            n_batches = 0
            
            for i in range(0, len(X_train_tensor), self.batch_size):
                batch_X = X_train_tensor[i:i+self.batch_size]
                batch_y = y_train_tensor[i:i+self.batch_size]
                
                optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                break
        
        # Restore best model
        if 'best_model_state' in locals():
            self.model.load_state_dict(best_model_state)
        
        self.is_fitted = True
        
        # Validation predictions
        self.model.eval()
        with torch.no_grad():
            val_outputs = self.model(X_val_tensor)
            val_proba = F.softmax(val_outputs, dim=1).cpu().numpy()
            
            if output_dim == 2:
                val_proba = val_proba[:, 1]  # Binary classification
        
        return {
            'model': self.model,
            'val_proba': val_proba,
            'training_samples': len(y_train_proc)
        }
    
    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on test data."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
            
        X_test_scaled = self.scaler.transform(X_test)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            proba = F.softmax(outputs, dim=1).cpu().numpy()
            predictions = np.argmax(proba, axis=1)
            
            # Convert back to original labels
            y_pred = self.label_encoder.inverse_transform(predictions)
            
            # For binary classification, return probability of positive class
            # For multi-class, return full probability matrix
            if proba.shape[1] == 2:
                y_proba = proba[:, 1]
            else:
                y_proba = proba
        
        return y_pred, y_proba
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'algorithm': 'SAINT',
            'preprocessing': 'StandardScaler + SMOTE',
            'hyperparameters': self.config,
            'architecture': f'SAINT with {self.depth} layers, dim={self.dim}, heads={self.heads}'
        }