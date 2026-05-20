"""
TabM (Tabular Deep Learning with Parameter-Efficient Ensembling) implementation using official model.

This implementation uses the official TabM model from ICLR 2025.
TabM efficiently imitates an ensemble of MLPs through parallel training and weight sharing.

Paper: "TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling" (ICLR 2025)
GitHub: https://github.com/yandex-research/tabm
"""

import numpy as np
import warnings
import time
import random
from typing import Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

# Try to import official TabM
try:
    import torch
    import torch.nn as nn
    from tabm import TabM
    TABM_AVAILABLE = True
except ImportError:
    TABM_AVAILABLE = False


class TabMClassifier_ML:
    """
    Official TabM wrapper for tabular classification.
    
    TabM (Tabular Deep Learning with Parameter-Efficient Ensembling) is a simple and powerful
    architecture that efficiently imitates an ensemble of MLPs through:
    - Parallel training of multiple MLPs
    - Weight sharing between MLPs for efficiency
    
    This is the original implementation without any fallback models.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize TabM classifier with official implementation.
        
        Parameters:
        -----------
        learning_rate : float, default=1e-3
            Learning rate for optimizer
        batch_size : int, default=256
            Batch size for training
        max_epochs : int, default=50
            Maximum number of training epochs
        patience : int, default=10
            Early stopping patience
        n_blocks : int, default=2
            Number of MLP blocks
        d_block : int, default=128
            Block dimension
        dropout : float, default=0.1
            Dropout rate
        k : int, default=32
            Ensemble size (number of parallel predictions)
        random_state : int, default=42
            Random seed for reproducibility
        device : str, default='auto'
            Device to use ('cuda', 'cpu', or 'auto')
        """
        if not TABM_AVAILABLE:
            raise ImportError(
                "TabM package is not available. Please install it with:\n"
                "pip install tabm rtdl_num_embeddings"
            )
        
        # TabM configuration parameters
        self.learning_rate = kwargs.get('learning_rate', 1e-3)
        self.batch_size = kwargs.get('batch_size', 256)
        self.max_epochs = kwargs.get('max_epochs', 50)
        self.patience = kwargs.get('patience', 10)
        self.n_blocks = kwargs.get('n_blocks', 2)
        self.d_block = kwargs.get('d_block', 128)
        self.dropout = kwargs.get('dropout', 0.1)
        self.k = kwargs.get('k', 32)
        self.random_state = kwargs.get('random_state', 42)
        
        # Device configuration
        device_arg = kwargs.get('device', 'auto')
        if device_arg == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device_arg
        
        # Data preprocessing components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')
        
        # Model state tracking
        self.model = None
        self.is_fitted = False
        self.n_features = None
        self.n_classes = None
        self.training_samples = None
        self.fit_time = None
        
        # Set random seeds for reproducibility
        self._set_seed(self.random_state)
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def preprocess_data(self, X_train, y_train, X_val, X_test):
        """
        Preprocess data for TabM model.
        
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
        # Handle missing values with median imputation
        X_train_clean = self.imputer.fit_transform(X_train)
        X_val_clean = self.imputer.transform(X_val)
        X_test_clean = self.imputer.transform(X_test)
        
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train_clean)
        X_val_scaled = self.scaler.transform(X_val_clean)
        X_test_scaled = self.scaler.transform(X_test_clean)
        
        # Encode labels to integers
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # Store dataset information
        self.n_features = X_train_scaled.shape[1]
        self.n_classes = len(np.unique(y_train_encoded))
        self.training_samples = len(X_train_scaled)
        
        return X_train_scaled, y_train_encoded, X_val_scaled, X_test_scaled
    
    def train(self, X_train, y_train, X_val, y_val):
        """
        Train TabM model.
        
        Parameters:
        -----------
        X_train : array-like of shape (n_samples, n_features)
            Training features
        y_train : array-like of shape (n_samples,)
            Training labels
        X_val : array-like of shape (n_val_samples, n_features)
            Validation features
        y_val : array-like of shape (n_val_samples,)
            Validation labels
        
        Returns:
        --------
        result : dict
            Training results including validation predictions
        """
        # Preprocess all data
        X_train_processed, y_train_processed, X_val_processed, _ = self.preprocess_data(
            X_train, y_train, X_val, X_val
        )
        
        # Encode validation labels
        y_val_encoded = self.label_encoder.transform(y_val)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_processed).to(self.device)
        y_train_tensor = torch.LongTensor(y_train_processed).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_processed).to(self.device)
        y_val_tensor = torch.LongTensor(y_val_encoded).to(self.device)
        
        # Initialize TabM model
        start_time = time.time()
        
        # Create TabM model
        # For classification, d_out is the number of classes
        # Pass backbone parameters directly as keyword arguments
        self.model = TabM.make(
            n_num_features=self.n_features,
            cat_cardinalities=[],  # No categorical features (already encoded)
            d_out=self.n_classes,  # Number of output classes
            k=self.k,
            n_blocks=self.n_blocks,
            d_block=self.d_block,
            dropout=self.dropout,
        ).to(self.device)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        # Loss function
        loss_fn = nn.CrossEntropyLoss()
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            n_batches = 0
            
            # Create batches
            n_train = len(X_train_tensor)
            indices = torch.randperm(n_train)
            
            for i in range(0, n_train, self.batch_size):
                batch_indices = indices[i:min(i + self.batch_size, n_train)]
                X_batch = X_train_tensor[batch_indices]
                y_batch = y_train_tensor[batch_indices]
                
                optimizer.zero_grad()
                
                # Forward pass - TabM outputs (batch_size, k, n_classes)
                # We need to reshape for loss computation
                outputs = self.model(X_batch, None)  # None for categorical features
                
                # Reshape: (batch_size, k, n_classes) -> (batch_size * k, n_classes)
                batch_size_actual = outputs.shape[0]
                k_size = outputs.shape[1]
                n_classes_out = outputs.shape[2]
                outputs_reshaped = outputs.reshape(batch_size_actual * k_size, n_classes_out)
                
                # Repeat labels k times: (batch_size,) -> (batch_size * k,)
                y_batch_repeated = y_batch.unsqueeze(1).repeat(1, k_size).reshape(-1)
                
                # Compute loss
                loss = loss_fn(outputs_reshaped, y_batch_repeated)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = train_loss / n_batches if n_batches > 0 else 0
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor, None)
                
                # Reshape for loss computation
                batch_size_val = val_outputs.shape[0]
                k_size = val_outputs.shape[1]
                n_classes_out = val_outputs.shape[2]
                val_outputs_reshaped = val_outputs.reshape(batch_size_val * k_size, n_classes_out)
                y_val_repeated = y_val_tensor.unsqueeze(1).repeat(1, k_size).reshape(-1)
                
                val_loss = loss_fn(val_outputs_reshaped, y_val_repeated).item()
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                break
        
        self.fit_time = time.time() - start_time
        self.is_fitted = True
        
        # Get validation predictions (average over ensemble)
        self.model.eval()
        with torch.no_grad():
            val_outputs = self.model(X_val_tensor, None)
            # Average predictions over k ensemble members: (batch, k, classes) -> (batch, classes)
            val_proba_tensor = torch.softmax(val_outputs, dim=2).mean(dim=1)
            val_proba = val_proba_tensor.cpu().numpy()
        
        return {
            'val_proba': val_proba,
            'training_samples': self.training_samples,
            'fit_time': self.fit_time,
            'model_info': self.get_model_info()
        }
    
    def predict(self, X_test):
        """
        Make predictions using fitted TabM model.
        
        Parameters:
        -----------
        X_test : array-like of shape (n_samples, n_features)
            Test features
        
        Returns:
        --------
        y_pred : array-like
            Predicted class labels
        y_proba : array-like
            Predicted class probabilities
        """
        if not self.is_fitted:
            raise ValueError("TabM model must be fitted before making predictions")
        
        # Preprocess test data using fitted preprocessors
        X_test_clean = self.imputer.transform(X_test)
        X_test_scaled = self.scaler.transform(X_test_clean)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            # Get predictions
            outputs = self.model(X_test_tensor, None)
            
            # Average predictions over k ensemble members
            # outputs shape: (batch, k, n_classes)
            probabilities_tensor = torch.softmax(outputs, dim=2).mean(dim=1)
            probabilities = probabilities_tensor.cpu().numpy()
            
            # Get class predictions
            predictions_encoded = np.argmax(probabilities, axis=1)
            
            # Convert back to original label encoding
            y_pred = self.label_encoder.inverse_transform(predictions_encoded)
            
            return y_pred, probabilities
    
    def get_model_info(self):
        """
        Get comprehensive model information and configuration.
        
        Returns:
        --------
        info : dict
            Model information, parameters, and metadata
        """
        return {
            'algorithm': 'TabM',
            'type': 'Parameter-Efficient Ensemble',
            'description': 'Efficient ensemble of MLPs with parallel training and weight sharing',
            'paper': 'TabM: Advancing Tabular Deep Learning With Parameter-Efficient Ensembling (ICLR 2025)',
            'github': 'https://github.com/yandex-research/tabm',
            'preprocessing': 'StandardScaler + SimpleImputer + LabelEncoder',
            'parameters': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'max_epochs': self.max_epochs,
                'patience': self.patience,
                'n_blocks': self.n_blocks,
                'd_block': self.d_block,
                'dropout': self.dropout,
                'k': self.k,
                'random_state': self.random_state,
                'device': self.device
            },
            'model_state': {
                'is_fitted': self.is_fitted,
                'n_features': self.n_features,
                'n_classes': self.n_classes,
                'training_samples': self.training_samples,
                'fit_time': self.fit_time
            },
            'advantages': [
                'Efficient ensemble through weight sharing',
                'Parallel training of ensemble members',
                'Strong performance on tabular data',
                'Parameter-efficient architecture',
                'Automatic ensemble averaging',
                'State-of-the-art ICLR 2025 model'
            ],
            'limitations': [
                'Requires PyTorch and tabm package',
                'More complex than single MLP',
                'Needs GPU for large datasets'
            ],
            'recommended_use_cases': [
                'Tabular classification tasks',
                'When ensemble methods are beneficial',
                'Medium to large datasets',
                'When computational efficiency matters'
            ]
        }

