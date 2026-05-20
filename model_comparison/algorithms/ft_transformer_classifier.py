"""
FT-Transformer implementation using official rtdl library.

Based on "Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., 2021)
Paper: https://arxiv.org/abs/2106.11959
Official Implementation: https://github.com/yandex-research/rtdl

This implementation uses the official rtdl library, not a fallback model.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import official rtdl library
try:
    import rtdl
    RTDL_AVAILABLE = True
except ImportError as e:
    RTDL_AVAILABLE = False
    _IMPORT_ERROR = str(e)


class FTTransformerClassifier_ML:
    """
    Official FT-Transformer wrapper for tabular classification.
    
    FT-Transformer (Feature Tokenizer + Transformer) is a state-of-the-art
    deep learning model for tabular data from Yandex Research.
    
    This is NOT a fallback model - requires rtdl library to be installed.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize FT-Transformer classifier with official rtdl library.
        
        Parameters:
        -----------
        n_blocks : int, default=2
            Number of Transformer blocks
        d_token : int, default=192
            Token dimension (embedding dimension)
        n_heads : int, default=8
            Number of attention heads
        attention_dropout : float, default=0.2
            Dropout rate for attention
        ffn_d_hidden : int, default=None
            Hidden dimension of FFN (if None, 4 * d_token)
        ffn_dropout : float, default=0.1
            Dropout rate for FFN
        residual_dropout : float, default=0.0
            Dropout rate for residual connections
        learning_rate : float, default=1e-4
            Learning rate for optimizer
        batch_size : int, default=256
            Batch size for training
        max_epochs : int, default=100
            Maximum number of training epochs
        patience : int, default=10
            Early stopping patience
        random_state : int, default=42
            Random seed
        """
        if not RTDL_AVAILABLE:
            error_msg = "rtdl (Revisiting Tabular Deep Learning) package is not available."
            if '_IMPORT_ERROR' in globals():
                error_msg += f"\nImport error: {_IMPORT_ERROR}"
            error_msg += "\nPlease install it with:\n"
            error_msg += "pip install rtdl"
            raise ImportError(error_msg)
        
        # FT-Transformer architecture parameters
        self.n_blocks = kwargs.get('n_blocks', 2)
        self.d_token = kwargs.get('d_token', 192)
        self.n_heads = kwargs.get('n_heads', 8)
        self.attention_dropout = kwargs.get('attention_dropout', 0.2)
        self.ffn_d_hidden = kwargs.get('ffn_d_hidden', None)
        self.ffn_dropout = kwargs.get('ffn_dropout', 0.1)
        self.residual_dropout = kwargs.get('residual_dropout', 0.0)
        
        # Training parameters
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        self.batch_size = kwargs.get('batch_size', 256)
        self.max_epochs = kwargs.get('max_epochs', 100)
        self.patience = kwargs.get('patience', 10)
        self.weight_decay = kwargs.get('weight_decay', 1e-5)
        self.random_state = kwargs.get('random_state', 42)
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')
        self.is_fitted = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seeds
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        
        # Store config
        self.config = kwargs
    
    def _build_model(self, n_features: int, n_classes: int):
        """
        Build FT-Transformer model using official rtdl library.
        
        Parameters:
        -----------
        n_features : int
            Number of input features
        n_classes : int
            Number of output classes
        
        Returns:
        --------
        model : rtdl.FTTransformer
            FT-Transformer model
        """
        # Calculate FFN hidden dimension
        ffn_d_hidden = self.ffn_d_hidden if self.ffn_d_hidden is not None else self.d_token * 4
        
        # Build FT-Transformer using make_baseline method (requires all parameters)
        model = rtdl.FTTransformer.make_baseline(
            n_num_features=n_features,
            cat_cardinalities=[],  # No categorical features (already encoded)
            d_out=n_classes,
            d_token=self.d_token,
            n_blocks=self.n_blocks,
            attention_dropout=self.attention_dropout,
            ffn_d_hidden=ffn_d_hidden,
            ffn_dropout=self.ffn_dropout,
            residual_dropout=self.residual_dropout  # Now supported in make_baseline
        )
        
        return model
    
    def preprocess_data(self, X_train, y_train, X_val, X_test):
        """
        Preprocess data for FT-Transformer.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        X_val : array-like
            Validation features
        X_test : array-like
            Test features
        
        Returns:
        --------
        Tuple of preprocessed arrays
        """
        # Handle missing values
        X_train_clean = self.imputer.fit_transform(X_train)
        X_val_clean = self.imputer.transform(X_val)
        X_test_clean = self.imputer.transform(X_test)
        
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train_clean)
        X_val_scaled = self.scaler.transform(X_val_clean)
        X_test_scaled = self.scaler.transform(X_test_clean)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        return X_train_scaled, y_train_encoded, X_val_scaled, X_test_scaled
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """
        Train FT-Transformer model using official rtdl library.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        X_val : np.ndarray
            Validation features
        y_val : np.ndarray
            Validation labels
        
        Returns:
        --------
        dict
            Training results
        """
        # Preprocess data
        X_train_proc, y_train_proc, X_val_proc, _ = self.preprocess_data(
            X_train, y_train, X_val, X_val
        )
        
        # Get dataset info
        n_features = X_train_proc.shape[1]
        n_classes = len(np.unique(y_train_proc))
        
        # Build model
        self.model = self._build_model(n_features, n_classes)
        self.model = self.model.to(self.device)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_proc).to(self.device)
        y_train_tensor = torch.LongTensor(y_train_proc).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_proc).to(self.device)
        y_val_encoded = self.label_encoder.transform(y_val)
        y_val_tensor = torch.LongTensor(y_val_encoded).to(self.device)
        
        # Setup optimizer and loss
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            self.model.train()
            
            # Mini-batch training
            n_samples = len(X_train_tensor)
            indices = torch.randperm(n_samples)
            
            for i in range(0, n_samples, self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                X_batch = X_train_tensor[batch_indices]
                y_batch = y_train_tensor[batch_indices]
                
                # Forward pass
                optimizer.zero_grad()
                logits = self.model(X_batch, None)  # No categorical features
                loss = criterion(logits, y_batch)
                
                # Backward pass
                loss.backward()
                optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val_tensor, None)
                val_loss = criterion(val_logits, y_val_tensor).item()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                break
        
        self.is_fitted = True
        
        # Get validation predictions
        self.model.eval()
        with torch.no_grad():
            val_logits = self.model(X_val_tensor, None)
            val_proba = torch.softmax(val_logits, dim=1).cpu().numpy()
        
        # For binary classification, return positive class probability
        if n_classes == 2:
            val_proba = val_proba[:, 1]
        
        return {
            'val_proba': val_proba,
            'training_samples': len(X_train_proc),
            'n_epochs': epoch + 1,
            'best_val_loss': best_val_loss,
            'model_info': self.get_model_info()
        }
    
    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using trained FT-Transformer.
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        
        Returns:
        --------
        y_pred : np.ndarray
            Predicted labels
        y_proba : np.ndarray
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Preprocess test data
        X_test_clean = self.imputer.transform(X_test)
        X_test_scaled = self.scaler.transform(X_test_clean)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_test_tensor, None)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
        
        # Convert back to original labels
        y_pred = self.label_encoder.inverse_transform(predictions)
        
        # For binary classification, return positive class probability
        # For multi-class, return full probability matrix
        n_classes = proba.shape[1]
        if n_classes == 2:
            y_proba = proba[:, 1]
        else:
            y_proba = proba
        
        return y_pred, y_proba
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get FT-Transformer model information.
        
        Returns:
        --------
        dict
            Model information
        """
        return {
            'algorithm': 'FT-Transformer',
            'type': 'Feature Tokenizer + Transformer (Official)',
            'description': 'State-of-the-art Transformer for tabular data with feature tokenization',
            'paper': 'Revisiting Deep Learning Models for Tabular Data (Gorishniy et al., 2021)',
            'github': 'https://github.com/yandex-research/rtdl',
            'library': 'rtdl (official implementation)',
            'preprocessing': 'SimpleImputer + StandardScaler + LabelEncoder',
            'parameters': {
                'n_blocks': self.n_blocks,
                'd_token': self.d_token,
                'n_heads': self.n_heads,
                'attention_dropout': self.attention_dropout,
                'ffn_d_hidden': self.ffn_d_hidden,
                'ffn_dropout': self.ffn_dropout,
                'residual_dropout': self.residual_dropout,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'max_epochs': self.max_epochs,
                'patience': self.patience,
                'weight_decay': self.weight_decay,
                'random_state': self.random_state
            },
            'model_state': {
                'is_fitted': self.is_fitted,
                'device': str(self.device)
            },
            'advantages': [
                'State-of-the-art for tabular data',
                'Feature tokenization for better representation',
                'Self-attention mechanism',
                'Official Yandex Research implementation',
                'GPU acceleration',
                'Proven performance on benchmarks'
            ],
            'limitations': [
                'Requires more data than tree-based models',
                'Slower training than GBM models',
                'Needs GPU for large datasets',
                'More hyperparameters to tune'
            ],
            'recommended_use_cases': [
                'Medium to large tabular datasets',
                'When deep learning is preferred',
                'GPU resources available',
                'Seeking state-of-the-art performance'
            ]
        }


# Alias for backward compatibility
FTTransformerClassifier = FTTransformerClassifier_ML
