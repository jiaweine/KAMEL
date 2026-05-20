"""
CatBoost Classifier for ML comparison experiments.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class CatBoostClassifier_ML:
    """CatBoost wrapper for ML comparison framework"""
    
    def __init__(self, **kwargs):
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not installed. Please install it with: pip install catboost")
        
        self.random_state = kwargs.get('random_state', 42)
        self.kwargs = kwargs
        
        # SMOTE for handling imbalanced datasets
        self.sampler = SMOTE(random_state=self.random_state)
        
        # Model will be initialized during training
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train CatBoost model"""
        try:
            # Convert to numpy arrays
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Handle categorical features (CatBoost can handle them natively)
            # For now, we'll treat all features as numerical
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Encode labels
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            
            # Apply SMOTE for class balancing if needed
            unique_classes = np.unique(y_train_encoded)
            if len(unique_classes) > 1:
                try:
                    X_train_balanced, y_train_balanced = self.sampler.fit_resample(X_train_scaled, y_train_encoded)
                except Exception:
                    # If SMOTE fails, use original data
                    X_train_balanced, y_train_balanced = X_train_scaled, y_train_encoded
            else:
                X_train_balanced, y_train_balanced = X_train_scaled, y_train_encoded
            
            # Prepare model parameters
            model_params = self.kwargs.copy()
            
            # Remove non-CatBoost parameters
            model_params.pop('random_state', None)
            
            # Set CatBoost-specific parameters
            catboost_params = {
                'iterations': model_params.get('n_estimators', 100),
                'learning_rate': model_params.get('learning_rate', 0.1),
                'depth': model_params.get('max_depth', 6),
                'l2_leaf_reg': model_params.get('reg_lambda', 3.0),
                'random_seed': self.random_state,
                'logging_level': 'Silent',  # Suppress output
                'allow_writing_files': False,  # Don't write files
                'thread_count': -1,  # Use all available threads
                'task_type': 'CPU'  # Use CPU by default
            }
            
            # Handle class imbalance
            if len(unique_classes) > 1:
                class_counts = np.bincount(y_train_balanced)
                if len(class_counts) > 1:
                    # Calculate class weights
                    total_samples = len(y_train_balanced)
                    n_classes = len(class_counts)
                    class_weights = {i: total_samples / (n_classes * count) for i, count in enumerate(class_counts)}
                    catboost_params['class_weights'] = class_weights
            
            # Initialize and train CatBoost model
            self.model = CatBoostClassifier(**catboost_params)
            
            # Prepare validation data if available
            eval_set = None
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                y_val_encoded = self.label_encoder.transform(y_val)
                eval_set = [(X_val_scaled, y_val_encoded)]
                catboost_params['early_stopping_rounds'] = model_params.get('patience', 10)
            
            # Train the model
            if eval_set:
                self.model.fit(
                    X_train_balanced, 
                    y_train_balanced,
                    eval_set=eval_set,
                    verbose=False
                )
            else:
                self.model.fit(X_train_balanced, y_train_balanced)
            
            self.is_fitted = True
            self._training_samples = len(X_train)  # Store for get_model_info
            
            # Prepare validation predictions if validation data is provided
            val_proba = None
            if X_val is not None:
                try:
                    val_proba = self.predict_proba(X_val)
                except Exception:
                    val_proba = None
            
            return {
                'val_proba': val_proba,
                'training_samples': len(X_train)
            }
            
        except Exception as e:
            warnings.warn(f"CatBoost training failed: {str(e)}")
            raise e
    
    def predict(self, X_test):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_test = np.array(X_test)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get predictions
        y_pred_encoded = self.model.predict(X_test_scaled)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        # Get prediction probabilities
        y_proba = self.model.predict_proba(X_test_scaled)
        
        return y_pred, y_proba
    
    def predict_proba(self, X_test):
        """Predict class probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_test = np.array(X_test)
        X_test_scaled = self.scaler.transform(X_test)
        
        return self.model.predict_proba(X_test_scaled)
    
    def get_model_info(self):
        """Get model information"""
        if not self.is_fitted:
            return {
                'name': 'CatBoost Classifier',
                'status': 'not_fitted'
            }
        
        # Get model parameters
        model_params = self.model.get_params() if hasattr(self.model, 'get_params') else {}
        
        return {
            'name': 'CatBoost Classifier',
            'algorithm': 'Gradient Boosting',
            'library': 'CatBoost',
            'parameters': {
                'iterations': model_params.get('iterations', 'unknown'),
                'learning_rate': model_params.get('learning_rate', 'unknown'),
                'depth': model_params.get('depth', 'unknown'),
                'l2_leaf_reg': model_params.get('l2_leaf_reg', 'unknown'),
                'random_seed': model_params.get('random_seed', 'unknown')
            },
            'features': {
                'handles_categorical': True,
                'handles_missing': True,
                'supports_gpu': True,
                'early_stopping': True
            },
            'status': 'fitted',
            'training_samples': getattr(self, '_training_samples', 'unknown')
        }


def train_algorithm(X_train, y_train, X_val, y_val, algorithm_config, device='cpu'):
    """Train CatBoost algorithm"""
    
    # Get configuration
    config = algorithm_config.get('catboost', {})
    
    # Initialize model
    model = CatBoostClassifier_ML(**config)
    
    # Train model
    train_result = model.train(X_train, y_train, X_val, y_val)
    
    # Test predictions
    y_pred, y_proba = model.predict(X_val)
    
    return model, train_result, y_pred, y_proba
