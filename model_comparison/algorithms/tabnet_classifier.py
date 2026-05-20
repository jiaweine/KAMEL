"""
TabNet implementation for tabular classification.
Based on the paper: "TabNet: Attentive Interpretable Tabular Learning"
Fixed version with proper dimension handling and class weighting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, Any, Tuple, Optional, List
import warnings
import random
warnings.filterwarnings('ignore')


class GhostBatchNorm(nn.Module):
    """Ghost Batch Normalization for TabNet."""
    
    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.01):
        super(GhostBatchNorm, self).__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(input_dim, momentum=momentum)
        
    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]
        return torch.cat(res, dim=0)


class GLU(nn.Module):
    """Gated Linear Unit."""
    
    def __init__(self, input_dim, output_dim, virtual_batch_size=128, momentum=0.02):
        super(GLU, self).__init__()
        self.output_dim = output_dim
        self.fc = nn.Linear(input_dim, 2 * output_dim, bias=False)
        self.bn = GhostBatchNorm(2 * output_dim, virtual_batch_size, momentum=momentum)
        
    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        out = torch.mul(x[:, :self.output_dim], torch.sigmoid(x[:, self.output_dim:]))
        return out


class AttentiveTransformer(nn.Module):
    """Attentive transformer for feature selection."""
    
    def __init__(self, input_dim, output_dim, virtual_batch_size=128, momentum=0.02, mask_type="sparsemax"):
        super(AttentiveTransformer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.bn = GhostBatchNorm(output_dim, virtual_batch_size, momentum=momentum)
        self.selector = nn.Linear(output_dim, input_dim, bias=False)
        self.mask_type = mask_type
        
    def forward(self, priors, processed_feat):
        x = self.fc(processed_feat)
        x = self.bn(x)
        # Don't multiply here - dimensions don't match (output_dim vs input_dim)
        # The multiplication should happen after selector
        x = self.selector(x)
        x = torch.mul(x, priors)  # Now both are [batch_size, input_dim]
        if self.mask_type == "sparsemax":
            x = self.sparsemax(x)
        else:
            x = F.softmax(x, dim=-1)
        return x
    
    def sparsemax(self, x):
        """Sparsemax activation function."""
        original_size = x.size()
        x = x.view(-1, x.size(-1))
        
        dim = -1
        number_of_logits = x.size(dim)
        
        # Translate for numerical stability
        x_max, _ = torch.max(x, dim=dim, keepdim=True)
        x = x - x_max
        
        # Sort x in descending order
        zs = torch.sort(x, dim=dim, descending=True)[0]
        range_tensor = torch.arange(start=1, end=number_of_logits + 1, dtype=x.dtype, device=x.device)
        range_tensor = range_tensor.view(1, -1)
        range_tensor = range_tensor.repeat(x.size(0), 1)
        
        # Determine sparsity of projection
        bound = 1 + range_tensor * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(x.type())
        k = torch.max(is_gt * range_tensor, dim, keepdim=True)[0]
        
        # Compute threshold function
        zs_sparse = zs * is_gt
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.repeat(1, number_of_logits)
        
        # Sparsemax
        self.output = torch.max(torch.zeros_like(x), x - taus)
        
        # Reshape back
        output = self.output.view(original_size)
        return output


class TabNetLayer(nn.Module):
    """Single TabNet decision step."""
    
    def __init__(self, input_dim, output_dim, n_d, n_a, n_shared, n_ind, 
                 original_features, vbs=128, momentum=0.02):
        super(TabNetLayer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.original_features = original_features
        
        # Shared layers
        if n_shared > 0:
            shared_layers = []
            shared_layers.append(nn.Linear(input_dim, n_shared, bias=False))
            shared_layers.append(GhostBatchNorm(n_shared, vbs, momentum=momentum))
            shared_layers.append(nn.ReLU())
            self.shared_layers = nn.Sequential(*shared_layers)
            shared_out = n_shared
        else:
            self.shared_layers = None
            shared_out = input_dim
            
        # Decision layers
        if n_ind > 0:
            decision_layers = []
            decision_layers.append(nn.Linear(shared_out, n_ind, bias=False))
            decision_layers.append(GhostBatchNorm(n_ind, vbs, momentum=momentum))
            decision_layers.append(nn.ReLU())
            decision_layers.append(nn.Linear(n_ind, n_d, bias=False))
            decision_layers.append(GhostBatchNorm(n_d, vbs, momentum=momentum))
            decision_layers.append(nn.ReLU())
            self.decision_layers = nn.Sequential(*decision_layers)
        else:
            self.decision_layers = nn.Linear(shared_out, n_d, bias=False)
            
        # Attention layers - output to original_features dimension
        if n_ind > 0:
            attention_layers = []
            attention_layers.append(nn.Linear(shared_out, n_ind, bias=False))
            attention_layers.append(GhostBatchNorm(n_ind, vbs, momentum=momentum))
            attention_layers.append(nn.ReLU())
            attention_layers.append(nn.Linear(n_ind, self.original_features, bias=False))
            attention_layers.append(GhostBatchNorm(self.original_features, vbs, momentum=momentum))
            attention_layers.append(nn.ReLU())
            self.attention_layers = nn.Sequential(*attention_layers)
        else:
            self.attention_layers = nn.Linear(shared_out, self.original_features, bias=False)
            
    def forward(self, x, prior_scales):
        if self.shared_layers is not None:
            x = self.shared_layers(x)
            
        # Decision output
        decision_out = self.decision_layers(x)
        
        # Attention output
        attention_out = self.attention_layers(x)
        
        return decision_out, attention_out


class TabNetModel(nn.Module):
    """TabNet model implementation."""
    
    def __init__(self, input_dim, output_dim, n_d=8, n_a=8, n_steps=3, 
                 gamma=1.3, n_independent=2, n_shared=2, epsilon=1e-15,
                 virtual_batch_size=128, momentum=0.02, mask_type="sparsemax"):
        super(TabNetModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.mask_type = mask_type
        
        # Feature transformer
        self.initial_bn = GhostBatchNorm(input_dim, virtual_batch_size, momentum)
        
        # TabNet layers
        self.steps = nn.ModuleList()
        self.att_transformers = nn.ModuleList()
        
        for step in range(n_steps):
            transformer = AttentiveTransformer(
                input_dim, n_a, virtual_batch_size, momentum, mask_type
            )
            self.att_transformers.append(transformer)
            
            layer = TabNetLayer(
                input_dim, output_dim, n_d, n_a, n_shared, n_independent,
                input_dim, virtual_batch_size, momentum
            )
            self.steps.append(layer)
        
        # Final mapping
        self.final_mapping = nn.Linear(n_d, output_dim, bias=False)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x):
        x = self.initial_bn(x)
        bs = x.size(0)
        
        # Initialize prior scales
        prior_scales = torch.ones(bs, self.input_dim).to(x.device)
        masks = []
        
        # Aggregate decision outputs
        output_agg = torch.zeros(bs, self.n_d).to(x.device)
        
        for step_i in range(self.n_steps):
            # Feature selection
            mask = self.att_transformers[step_i](prior_scales, x)
            masks.append(mask)
            
            # Apply mask
            masked_x = torch.mul(mask, x)
            
            # TabNet step
            decision_out, attention_out = self.steps[step_i](masked_x, prior_scales)
            
            # Aggregate
            output_agg += F.relu(decision_out)
            
            # Update prior scales
            prior_scales = torch.mul(self.gamma - mask, prior_scales)
        
        # Final output
        out = self.final_mapping(output_agg)
        
        return out, masks


class TabNetClassifier_ML:
    """TabNet classifier with proper dimension handling and class weighting."""
    
    def __init__(self, **kwargs):
        # TabNet specific parameters
        self.n_d = kwargs.get('n_d', 8)
        self.n_a = kwargs.get('n_a', 8) 
        self.n_steps = kwargs.get('n_steps', 3)
        self.gamma = kwargs.get('gamma', 1.3)
        self.lambda_sparse = kwargs.get('lambda_sparse', 1e-3)
        self.n_independent = kwargs.get('n_independent', 2)
        self.n_shared = kwargs.get('n_shared', 2)
        self.virtual_batch_size = kwargs.get('virtual_batch_size', 128)
        self.momentum = kwargs.get('momentum', 0.02)
        self.mask_type = kwargs.get('mask_type', 'sparsemax')
        
        # Training parameters
        self.max_epochs = kwargs.get('max_epochs', 100)
        self.patience = kwargs.get('patience', 10)
        self.learning_rate = kwargs.get('learning_rate', 2e-2)
        self.weight_decay = kwargs.get('weight_decay', 1e-4)
        self.batch_size = kwargs.get('batch_size', 256)
        
        self.random_state = kwargs.get('random_state', 42)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Data preprocessing
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')
        self.is_fitted = False
        
        # Model components
        self.model = None
        self.n_features = None
        self.n_classes = None
        self.class_weights = None
        
        # Store configuration
        self.config = kwargs
        
        # Set random seeds
        self._set_random_seeds()
        
    def _set_random_seeds(self):
        """Set random seeds for reproducibility."""
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        random.seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)
    
    def preprocess_data(self, X, y=None, fit=False):
        """Preprocess data for TabNet."""
        
        if fit:
            # Handle missing values
            X_processed = self.imputer.fit_transform(X)
            
            # Scale features
            X_processed = self.scaler.fit_transform(X_processed)
            
            self.n_features = X_processed.shape[1]
            
            if y is not None:
                # Encode labels
                y_processed = self.label_encoder.fit_transform(y)
                self.n_classes = len(np.unique(y_processed))
                
                # Compute class weights
                class_weights = compute_class_weight(
                    'balanced', 
                    classes=np.unique(y_processed), 
                    y=y_processed
                )
                self.class_weights = torch.FloatTensor(class_weights).to(self.device)
                
                return X_processed, y_processed
            else:
                return X_processed
        else:
            # Transform using fitted preprocessors
            X_processed = self.imputer.transform(X)
            X_processed = self.scaler.transform(X_processed)
            
            if y is not None:
                y_processed = self.label_encoder.transform(y)
                return X_processed, y_processed
            else:
                return X_processed
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train TabNet model."""
        
        # Preprocess data
        X_train_scaled, y_train_encoded = self.preprocess_data(X_train, y_train, fit=True)
        X_val_scaled, y_val_encoded = self.preprocess_data(X_val, y_val, fit=False)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_tensor = torch.LongTensor(y_train_encoded).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
        y_val_tensor = torch.LongTensor(y_val_encoded).to(self.device)
        
        # Create model
        self.model = TabNetModel(
            input_dim=self.n_features,
            output_dim=self.n_classes,
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum,
            mask_type=self.mask_type
        ).to(self.device)
        
        # Loss function with class weights
        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        
        # Optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        
        print(f"Training TabNet on {X_train.shape[0]} samples...")
        
        for epoch in range(self.max_epochs):
            # Training phase
            self.model.train()
            
            # Create batches
            n_batches = int(np.ceil(len(X_train_tensor) / self.batch_size))
            train_loss = 0.0
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, len(X_train_tensor))
                
                batch_X = X_train_tensor[start_idx:end_idx]
                batch_y = y_train_tensor[start_idx:end_idx]
                
                optimizer.zero_grad()
                
                # Forward pass
                logits, masks = self.model(batch_X)
                
                # Compute loss
                loss = criterion(logits, batch_y)
                
                # Add sparsity regularization
                if self.lambda_sparse > 0:
                    sparsity_loss = 0
                    for mask in masks:
                        sparsity_loss += torch.mean(torch.sum(mask, dim=1))
                    loss += self.lambda_sparse * sparsity_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_logits, _ = self.model(X_val_tensor)
                val_loss = criterion(val_logits, y_val_tensor)
                val_predictions = torch.argmax(val_logits, dim=1)
                val_accuracy = (val_predictions == y_val_tensor).float().mean().item()
            
            # Early stopping
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                patience_counter = 0
                # Save best model state
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss/n_batches:.4f}, Val Accuracy = {val_accuracy:.4f}")
            
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        self.is_fitted = True
        
        # Validation predictions for return
        self.model.eval()
        with torch.no_grad():
            val_logits, _ = self.model(X_val_tensor)
            val_proba = F.softmax(val_logits, dim=1).cpu().numpy()
        
        return {
            'model': self.model,
            'val_proba': val_proba[:, 1] if self.n_classes == 2 else val_proba,
            'training_samples': len(y_train),
            'best_val_accuracy': best_val_acc
        }
    
    def predict(self, X_test):
        """Make predictions on test data."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
            
        # Preprocess test data
        X_test_scaled = self.preprocess_data(X_test, fit=False)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(X_test_tensor)
            probabilities = F.softmax(logits, dim=1).cpu().numpy()
            predictions = np.argmax(probabilities, axis=1)
            
            # Convert back to original labels
            y_pred = self.label_encoder.inverse_transform(predictions)
            
        # Return probabilities
        if self.n_classes == 2:
            y_proba = probabilities[:, 1]
        else:
            y_proba = probabilities
        
        return y_pred, y_proba
    
    def get_model_info(self):
        """Get model information."""
        return {
            'algorithm': 'TabNet',
            'preprocessing': 'StandardScaler + ClassWeight',
            'hyperparameters': self.config,
            'architecture': f'TabNet: n_d={self.n_d}, n_a={self.n_a}, n_steps={self.n_steps}'
        }
