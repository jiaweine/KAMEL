"""
TabR (Tabular Deep Learning via Retrieval) Classifier Implementation

Based on official Yandex Research implementation:
Paper: "TabR: Unlocking the Power of Retrieval-Augmented Tabular Deep Learning" (NeurIPS 2023)
Authors: Yury Gorishniy, Ivan Rubachev, Nikolay Kartashev, Daniil Shlenskii, Akim Kotelnikov, Artem Babenko
Official Repository: https://github.com/yandex-research/tabular-dl-tabr
ArXiv: https://arxiv.org/abs/2307.14338

This implementation uses the OFFICIAL TabR code from Yandex Research - NOT a simplified version.
TabR is a retrieval-augmented deep learning model that achieves state-of-the-art performance on tabular data.

Key Features:
- Retrieval-based context mechanism using k-nearest neighbors
- Feature Tokenizer + Transformer architecture
- FAISS-based efficient similarity search
- Memory-efficient training mode

Requirements:
- torch
- numpy
- faiss-cpu or faiss-gpu
- delu
- Official TabR library (included in tabr_official/)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Add TabR official library to path
TABR_OFFICIAL_PATH = os.path.join(os.path.dirname(__file__), 'tabr_official')
if TABR_OFFICIAL_PATH not in sys.path:
    sys.path.insert(0, TABR_OFFICIAL_PATH)

# Check TabR availability
TABR_AVAILABLE = False
_IMPORT_ERROR = None

try:
    # Set PROJECT_DIR environment variable (required by TabR official code)
    # This must be set BEFORE importing any TabR modules
    if 'PROJECT_DIR' not in os.environ:
        os.environ['PROJECT_DIR'] = TABR_OFFICIAL_PATH
    
    # Create cache directory (required by TabR)
    cache_dir = os.path.join(TABR_OFFICIAL_PATH, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Import official TabR implementation
    import delu
    import faiss
    
    # Add TabR bin and lib to path
    tabr_bin_path = os.path.join(TABR_OFFICIAL_PATH, 'bin')
    tabr_lib_path = os.path.join(TABR_OFFICIAL_PATH, 'lib')
    for path in [tabr_bin_path, tabr_lib_path]:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    # Import TabR model class directly
    from tabr import Model as TabRModel  # type: ignore[import-not-found]
    
    # Import necessary utilities from lib (not actually needed for basic usage)
    # import deep as tabr_deep  # type: ignore[import-not-found]
    
    TABR_AVAILABLE = True
    print("✓ TabR (Official Yandex Implementation) loaded successfully!")
    
except Exception as e:
    _IMPORT_ERROR = str(e)
    print(f"✗ TabR loading failed: {e}")
    import traceback
    traceback.print_exc()
    TabRModel = None


class TabRClassifier_ML:
    """
    Official TabR (Tabular Retrieval) Classifier for KAMEL Framework
    
    TabR is a retrieval-augmented deep learning model that uses k-nearest neighbors 
    to provide context for predictions. It achieves state-of-the-art results on many
    tabular datasets.
    
    This is the OFFICIAL implementation from Yandex Research - NOT a fallback model.
    
    Architecture:
    1. Feature Encoder: Transforms input features into embedding space
    2. Retrieval Module: Uses FAISS to find k-nearest neighbors in training set
    3. Context Integration: Combines query with retrieved context via attention
    4. Predictor: Final classification head
    
    Parameters
    ----------
    d_main : int, default=128
        Main embedding dimension
    d_multiplier : float, default=2.0
        Hidden layer dimension multiplier (hidden_dim = d_main * d_multiplier)
    encoder_n_blocks : int, default=0
        Number of encoder blocks (0 means just linear layer)
    predictor_n_blocks : int, default=1
        Number of predictor blocks
    context_dropout : float, default=0.0
        Dropout rate for context attention
    dropout0 : float, default=0.0
        First dropout rate in MLP blocks
    dropout1 : float, default=0.0
        Second dropout rate in MLP blocks
    normalization : str, default='LayerNorm'
        Normalization type ('LayerNorm', 'BatchNorm1d', etc.)
    activation : str, default='ReLU'
        Activation function ('ReLU', 'GELU', etc.)
    context_size : int, default=96
        Number of nearest neighbors to retrieve
    learning_rate : float, default=1e-3
        Learning rate for Adam optimizer
    weight_decay : float, default=0.0
        Weight decay for optimizer
    batch_size : int, default=256
        Training batch size
    max_epochs : int, default=100
        Maximum training epochs
    patience : int, default=15
        Early stopping patience
    random_state : int, default=42
        Random seed for reproducibility
    device : str, default='auto'
        Device to use ('cuda', 'cpu', or 'auto')
    """
    
    def __init__(self, **kwargs):
        """Initialize TabR classifier with official implementation."""
        
        if not TABR_AVAILABLE:
            error_msg = (
                "TabR (Official Yandex Research) is not available.\n"
                f"Import error: {_IMPORT_ERROR}\n\n"
                "Required dependencies:\n"
                "  pip install torch faiss-cpu delu\n\n"
                "TabR official code should be in: tabr_official/\n"
                "Clone from: https://github.com/yandex-research/tabular-dl-tabr"
            )
            raise ImportError(error_msg)
        
        # Model architecture parameters
        self.d_main = kwargs.get('d_main', 128)
        self.d_multiplier = kwargs.get('d_multiplier', 2.0)
        self.encoder_n_blocks = kwargs.get('encoder_n_blocks', 0)
        self.predictor_n_blocks = kwargs.get('predictor_n_blocks', 1)
        self.context_dropout = kwargs.get('context_dropout', 0.0)
        self.dropout0 = kwargs.get('dropout0', 0.0)
        self.dropout1 = kwargs.get('dropout1', 0.0)
        self.normalization = kwargs.get('normalization', 'LayerNorm')
        self.activation = kwargs.get('activation', 'ReLU')
        
        # Retrieval parameters
        self.context_size = kwargs.get('context_size', 96)
        
        # Training parameters
        self.learning_rate = kwargs.get('learning_rate', 1e-3)
        self.weight_decay = kwargs.get('weight_decay', 0.0)
        self.batch_size = kwargs.get('batch_size', 256)
        self.max_epochs = kwargs.get('max_epochs', 100)
        self.patience = kwargs.get('patience', 15)
        self.random_state = kwargs.get('random_state', 42)
        
        # Device configuration
        # Force CPU to avoid FAISS GPU compatibility issues
        # faiss-cpu doesn't support GpuIndexFlatL2
        device = kwargs.get('device', 'cpu')
        if device == 'auto':
            self.device = 'cpu'  # Force CPU for FAISS compatibility
        else:
            self.device = device if device != 'cuda' else 'cpu'  # Convert cuda to cpu
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')
        self.is_fitted = False
        
        # Training data (needed for retrieval)
        self.X_train_encoded = None
        self.y_train_encoded = None
        
        # Set random seed
        self._set_seed(self.random_state)
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def _prepare_data(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                      fit: bool = False) -> tuple:
        """
        Prepare data for TabR model.
        
        Returns:
            X_dict: Dictionary with 'num' key for numerical features
            y_encoded: Encoded labels (if y is provided)
        """
        # Handle missing values
        if fit:
            X = self.imputer.fit_transform(X)
        else:
            X = self.imputer.transform(X)
        
        # Standardize features
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # TabR expects a dictionary format
        X_dict = {'num': X_tensor}
        
        if y is not None:
            if fit:
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = self.label_encoder.transform(y)
            
            y_tensor = torch.LongTensor(y_encoded).to(self.device)
            return X_dict, y_tensor
        
        return X_dict, None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> dict:
        """
        Train TabR model (KAMEL framework interface).
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data
        y_train : array-like of shape (n_samples,)
            Training labels
        X_val : array-like of shape (n_samples, n_features), optional
            Validation data
        y_val : array-like of shape (n_samples,), optional
            Validation labels
            
        Returns
        -------
        results : dict
            Dictionary with 'val_proba' and 'training_samples' keys
        """
        # Fit the model
        self.fit(X_train, y_train)
        
        # Get validation predictions if validation data is provided
        if X_val is not None and y_val is not None:
            val_proba = self.predict_proba(X_val)
        else:
            # No validation data, use dummy probabilities
            val_proba = np.zeros((1, 2))
        
        return {
            'val_proba': val_proba,
            'training_samples': len(y_train)
        }
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TabRClassifier_ML':
        """
        Train TabR model on data (sklearn interface).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target labels
            
        Returns
        -------
        self : TabRClassifier_ML
            Fitted classifier
        """
        # Validate input
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        
        n_samples, n_features = X.shape
        
        # Prepare data
        X_dict, y_tensor = self._prepare_data(X, y, fit=True)
        
        # Store training data for retrieval
        self.X_train_encoded = X_dict
        self.y_train_encoded = y_tensor
        
        # Get number of classes
        n_classes = len(np.unique(y))
        
        # Create validation split (15% as in KAMEL)
        val_size = int(0.15 * n_samples)
        indices = torch.randperm(n_samples, generator=torch.Generator().manual_seed(self.random_state))
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        X_train = {k: v[train_indices] for k, v in X_dict.items()}
        y_train = y_tensor[train_indices]
        X_val = {k: v[val_indices] for k, v in X_dict.items()}
        y_val = y_tensor[val_indices]
        
        # Initialize TabR model with official implementation
        self.model = TabRModel(
            n_num_features=n_features,
            n_bin_features=0,  # No binary features
            cat_cardinalities=[],  # No categorical features
            n_classes=n_classes,
            num_embeddings=None,  # No feature embeddings
            d_main=self.d_main,
            d_multiplier=self.d_multiplier,
            encoder_n_blocks=self.encoder_n_blocks,
            predictor_n_blocks=self.predictor_n_blocks,
            mixer_normalization='auto',
            context_dropout=self.context_dropout,
            dropout0=self.dropout0,
            dropout1=self.dropout1,
            normalization=self.normalization,
            activation=self.activation,
            memory_efficient=False,
            candidate_encoding_batch_size=None,
        )
        
        # Move model to the device (should already be CPU)
        self.model.to(self.device)
        
        # Optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.max_epochs):
            self.model.train()
            
            # Mini-batch training
            n_train = len(train_indices)
            epoch_losses = []
            
            for i in range(0, n_train, self.batch_size):
                batch_end = min(i + self.batch_size, n_train)
                batch_indices = train_indices[i:batch_end]
                
                X_batch = {k: X_dict[k][batch_indices] for k in X_dict}
                y_batch = y_tensor[batch_indices]
                
                optimizer.zero_grad()
                
                # Forward pass with TabR's retrieval mechanism
                # Use full training data as candidates
                logits = self.model(
                    x_=X_batch,
                    y=y_batch,
                    candidate_x_=X_train,
                    candidate_y=y_train,
                    context_size=min(self.context_size, len(y_train)),
                    is_train=True
                )
                
                # Handle output dimensions
                if logits.dim() == 1:
                    # Single value per sample - convert to 2D for binary classification
                    logits = torch.stack([1 - logits, logits], dim=1)
                elif logits.shape[-1] == 1:
                    # (batch, 1) - squeeze to (batch,) then expand to (batch, 2)
                    logits = logits.squeeze(-1)
                    logits = torch.stack([1 - logits, logits], dim=1)
                
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            # Validation
            if len(val_indices) > 0:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(
                        x_=X_val,
                        y=None,
                        candidate_x_=X_train,
                        candidate_y=y_train,
                        context_size=min(self.context_size, len(y_train)),
                        is_train=False
                    )
                    
                    # Handle output dimensions for validation
                    if val_logits.dim() == 1:
                        val_logits = torch.stack([1 - val_logits, val_logits], dim=1)
                    elif val_logits.shape[-1] == 1:
                        val_logits = val_logits.squeeze(-1)
                        val_logits = torch.stack([1 - val_logits, val_logits], dim=1)
                    
                    val_loss = criterion(val_logits, y_val).item()
                    
                    # Early stopping check
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model state
                        best_model_state = self.model.state_dict()
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
            if (epoch + 1) % 10 == 0:
                avg_train_loss = np.mean(epoch_losses)
                print(f"Epoch {epoch+1}/{self.max_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Restore best model
        if 'best_model_state' in locals():
            self.model.load_state_dict(best_model_state)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> tuple:
        """
        Predict class labels and probabilities for samples in X (KAMEL interface).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
            
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels
        y_proba : array-like of shape (n_samples, n_classes)
            Predicted class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get probabilities first
        y_proba = self.predict_proba(X)
        
        # Get predictions from probabilities
        y_pred_encoded = np.argmax(y_proba, axis=1)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return y_pred, y_proba
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
            
        Returns
        -------
        probas : array-like of shape (n_samples, n_classes)
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.asarray(X, dtype=np.float32)
        X_dict, _ = self._prepare_data(X, fit=False)
        
        self.model.eval()
        with torch.no_grad():
            # Predict in batches
            all_probas = []
            n_samples = X.shape[0]
            eval_batch_size = 1024
            
            for i in range(0, n_samples, eval_batch_size):
                batch_end = min(i + eval_batch_size, n_samples)
                X_batch = {k: v[i:batch_end] for k, v in X_dict.items()}
                
                logits = self.model(
                    x_=X_batch,
                    y=None,
                    candidate_x_=self.X_train_encoded,
                    candidate_y=self.y_train_encoded,
                    context_size=min(self.context_size, len(self.y_train_encoded)),
                    is_train=False
                )
                
                # Handle output dimensions
                if logits.dim() == 1:
                    # Single value per sample - convert to 2D
                    logits = torch.stack([1 - logits, logits], dim=1)
                elif logits.shape[-1] == 1:
                    logits = logits.squeeze(-1)
                    logits = torch.stack([1 - logits, logits], dim=1)
                
                probas = torch.softmax(logits, dim=1)
                all_probas.append(probas.cpu().numpy())
            
            probabilities = np.concatenate(all_probas, axis=0)
        
        return probabilities
    
    def get_params(self, deep=True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        return {
            'd_main': self.d_main,
            'd_multiplier': self.d_multiplier,
            'encoder_n_blocks': self.encoder_n_blocks,
            'predictor_n_blocks': self.predictor_n_blocks,
            'context_dropout': self.context_dropout,
            'dropout0': self.dropout0,
            'dropout1': self.dropout1,
            'normalization': self.normalization,
            'activation': self.activation,
            'context_size': self.context_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'max_epochs': self.max_epochs,
            'patience': self.patience,
            'random_state': self.random_state,
            'device': self.device,
        }
    
    def set_params(self, **params) -> 'TabRClassifier_ML':
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information (KAMEL framework interface)."""
        return {
            'algorithm': 'TabR',
            'model_name': 'TabR (Official Yandex Research)',
            'paper': 'TabR: Unlocking the Power of Retrieval-Augmented Tabular Deep Learning (NeurIPS 2023)',
            'authors': 'Gorishniy et al.',
            'hyperparameters': self.get_params(),
            'architecture': {
                'd_main': self.d_main,
                'd_multiplier': self.d_multiplier,
                'encoder_n_blocks': self.encoder_n_blocks,
                'predictor_n_blocks': self.predictor_n_blocks,
                'context_size': self.context_size
            }
        }


# Convenience function to create TabR classifier
def create_tabr_classifier(**kwargs) -> TabRClassifier_ML:
    """
    Create TabR classifier with specified parameters.
    
    Example:
    --------
    >>> clf = create_tabr_classifier(
    ...     d_main=128,
    ...     encoder_n_blocks=2,
    ...     context_size=96,
    ...     learning_rate=1e-3,
    ...     max_epochs=100
    ... )
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    """
    return TabRClassifier_ML(**kwargs)


if __name__ == "__main__":
    print("="*80)
    print("TabR (Tabular Retrieval) - Official Yandex Research Implementation")
    print("="*80)
    print("\nPaper: TabR: Unlocking the Power of Retrieval-Augmented Tabular Deep Learning")
    print("Authors: Gorishniy et al., NeurIPS 2023")
    print("Repository: https://github.com/yandex-research/tabular-dl-tabr")
    print("\nThis is the OFFICIAL implementation - NOT a simplified fallback!")
    print("="*80)
    
    if TABR_AVAILABLE:
        print("\n✓ TabR is available and ready to use!")
        print(f"✓ Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    else:
        print(f"\n✗ TabR is NOT available")
        print(f"✗ Error: {_IMPORT_ERROR}")

