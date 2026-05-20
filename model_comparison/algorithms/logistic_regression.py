"""
Logistic Regression implementation for ML comparison experiments.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


class LogisticRegressionClassifier:
    """
    Logistic Regression with standard preprocessing and oversampling.
    """
    
    def __init__(self, **kwargs):
        self.random_state = kwargs.get('random_state', 42)
        self.model = None
        self.scaler = StandardScaler()
        self.sampler = SMOTE(random_state=self.random_state)
        self.is_fitted = False
        # Store algorithm configuration
        self.config = kwargs
        
    def preprocess_data(self, X_train: np.ndarray, y_train: np.ndarray, 
                       X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Standard ML preprocessing: scaling and oversampling.
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply oversampling to training data
        X_train_resampled, y_train_resampled = self.sampler.fit_resample(X_train_scaled, y_train)
        
        return X_train_resampled, y_train_resampled, X_val_scaled, X_test_scaled
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """
        Train logistic regression model.
        """
        # Preprocess data
        X_train_proc, y_train_proc, X_val_proc, _ = self.preprocess_data(
            X_train, y_train, X_val, X_val
        )
        
        # Calculate class weights for remaining imbalance
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y_train_proc), 
            y=y_train_proc
        )
        class_weight_dict = dict(zip(np.unique(y_train_proc), class_weights))
        
        # Initialize model with configuration parameters
        model_params = self.config.copy()
        model_params['random_state'] = self.random_state
        self.model = LogisticRegression(**model_params)
        
        # Train model
        self.model.fit(X_train_proc, y_train_proc)
        self.is_fitted = True
        
        # Validation predictions
        val_proba_raw = self.model.predict_proba(X_val_proc)
        # Handle both binary and multi-class
        if val_proba_raw.shape[1] == 2:
            # Binary classification: return positive class probability
            val_proba = val_proba_raw[:, 1]
        else:
            # Multi-class: return full probability matrix
            val_proba = val_proba_raw
        
        return {
            'model': self.model,
            'val_proba': val_proba,
            'training_samples': len(y_train_proc)
        }
    
    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on test data.
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
            
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_proba_raw = self.model.predict_proba(X_test_scaled)
        
        # Handle both binary and multi-class
        if y_proba_raw.shape[1] == 2:
            # Binary classification: return positive class probability
            y_proba = y_proba_raw[:, 1]
        else:
            # Multi-class: return full probability matrix
            y_proba = y_proba_raw
        
        return y_pred, y_proba
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        """
        return {
            'algorithm': 'Logistic Regression',
            'preprocessing': 'StandardScaler + SMOTE',
            'hyperparameters': self.config
        }
