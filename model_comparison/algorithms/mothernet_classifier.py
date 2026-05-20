"""
MotherNet (A Foundational Hypernetwork for Tabular Classification) implementation using official model.

This implementation uses the official MotherNet model from ICLR 2025.
MotherNet is a hypernetwork foundational model (conditional neural process) for tabular data 
classification that creates a small neural network for each dataset.

Paper: "MotherNet: A Foundational Hypernetwork for Tabular Classification" (ICLR 2025)
GitHub: https://github.com/microsoft/ticl
"""

import numpy as np
import warnings
import time
import random
from typing import Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')

# Try to import official MotherNet
try:
    from ticl.prediction import MotherNetClassifier as OfficialMotherNetClassifier
    MOTHERNET_AVAILABLE = True
except ImportError as e:
    MOTHERNET_AVAILABLE = False
    _IMPORT_ERROR = str(e)


class MotherNetClassifier_ML:
    """
    Official MotherNet wrapper for tabular classification.
    
    MotherNet is a hypernetwork foundational model that creates a small neural network
    for each dataset through in-context learning. It uses a transformer-based architecture
    to process training data and generate task-specific model parameters.
    
    This is the original implementation without any fallback models.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize MotherNet classifier with official implementation.
        
        Parameters:
        -----------
        device : str, default='auto'
            Device to use ('cuda', 'cpu', or 'auto')
        random_state : int, default=42
            Random seed for reproducibility
        path : str, optional
            Path to a specific MotherNet checkpoint (uses default if None)
        scale : bool, default=True
            Whether to scale features internally
        inference_device : str, default='auto'
            Device for inference (can be different from training device)
        """
        if not MOTHERNET_AVAILABLE:
            error_msg = "ticl package (MotherNet) is not available."
            if '_IMPORT_ERROR' in globals():
                error_msg += f"\nImport error: {_IMPORT_ERROR}"
            error_msg += "\nPlease install it with:\n"
            error_msg += "git clone https://github.com/microsoft/ticl.git\n"
            error_msg += "cd ticl\n"
            error_msg += "pip install -e .\n"
            error_msg += "pip install gpytorch hyperopt category_encoders openml mlflow einops"
            raise ImportError(error_msg)
        
        # MotherNet configuration parameters
        self.random_state = kwargs.get('random_state', 42)
        
        # Device configuration
        device_arg = kwargs.get('device', 'auto')
        if device_arg == 'auto':
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device_arg
        
        inference_device_arg = kwargs.get('inference_device', 'auto')
        if inference_device_arg == 'auto':
            self.inference_device = self.device
        else:
            self.inference_device = inference_device_arg
        
        # Other parameters
        self.path = kwargs.get('path', None)  # Use default model if None
        self.scale = kwargs.get('scale', True)
        
        # Data preprocessing components
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
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
    
    def preprocess_data(self, X_train, y_train, X_val, X_test):
        """
        Preprocess data for MotherNet model.
        
        MotherNet handles scaling internally, but we need to:
        - Handle missing values
        - Ensure proper data types
        
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
        
        # Store dataset information
        self.n_features = X_train_clean.shape[1]
        self.n_classes = len(np.unique(y_train))
        self.training_samples = len(X_train_clean)
        
        return X_train_clean, y_train, X_val_clean, X_test_clean
    
    def train(self, X_train, y_train, X_val, y_val):
        """
        Fit MotherNet model using in-context learning.
        
        MotherNet doesn't require traditional training with gradient updates on the target dataset.
        Instead, it uses the training data to generate task-specific model parameters through
        its pre-trained hypernetwork.
        
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
        
        # Initialize MotherNet model
        start_time = time.time()
        
        try:
            # Initialize official MotherNet with configuration
            self.model = OfficialMotherNetClassifier(
                path=self.path,
                device=self.device,
                scale=self.scale,
                inference_device=self.inference_device
            )
            
            # Fit MotherNet (generates task-specific parameters via hypernetwork)
            self.model.fit(X_train_processed, y_train_processed)
            
            self.fit_time = time.time() - start_time
            self.is_fitted = True
            
            # Get validation predictions
            val_proba = self.model.predict_proba(X_val_processed)
            
            return {
                'val_proba': val_proba,
                'training_samples': self.training_samples,
                'fit_time': self.fit_time,
                'model_info': self.get_model_info()
            }
            
        except Exception as e:
            raise RuntimeError(f"MotherNet training failed: {str(e)}") from e
    
    def predict(self, X_test):
        """
        Make predictions using fitted MotherNet model.
        
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
            raise ValueError("MotherNet model must be fitted before making predictions")
        
        # Preprocess test data using fitted preprocessors
        X_test_clean = self.imputer.transform(X_test)
        
        try:
            # Get probability predictions
            probabilities = self.model.predict_proba(X_test_clean)
            
            # Get class predictions
            predictions = self.model.predict(X_test_clean)
            
            return predictions, probabilities
            
        except Exception as e:
            raise RuntimeError(f"MotherNet prediction failed: {str(e)}") from e
    
    def get_model_info(self):
        """
        Get comprehensive model information and configuration.
        
        Returns:
        --------
        info : dict
            Model information, parameters, and metadata
        """
        return {
            'algorithm': 'MotherNet',
            'type': 'Foundational Hypernetwork',
            'description': 'Hypernetwork that generates task-specific neural networks via in-context learning',
            'paper': 'MotherNet: A Foundational Hypernetwork for Tabular Classification (ICLR 2025)',
            'github': 'https://github.com/microsoft/ticl',
            'preprocessing': 'SimpleImputer (internal scaling handled by MotherNet)',
            'parameters': {
                'device': self.device,
                'inference_device': self.inference_device,
                'scale': self.scale,
                'path': self.path if self.path else 'default',
                'random_state': self.random_state
            },
            'model_state': {
                'is_fitted': self.is_fitted,
                'n_features': self.n_features,
                'n_classes': self.n_classes,
                'training_samples': self.training_samples,
                'fit_time': self.fit_time
            },
            'advantages': [
                'In-context learning capability',
                'No gradient updates needed on target dataset',
                'Generates task-specific networks',
                'Strong performance on small to medium datasets',
                'Fast inference',
                'Pre-trained foundational model',
                'Based on TabPFN architecture'
            ],
            'limitations': [
                'Supports up to 10 classes',
                'Optimal for datasets with <=500 features',
                'Requires pre-trained checkpoint',
                'Memory requirements for model loading'
            ],
            'recommended_use_cases': [
                'Small to medium tabular classification',
                'Few-shot learning scenarios',
                'When training time is critical',
                'Datasets with <=10 classes'
            ]
        }

