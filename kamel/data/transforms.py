"""
Multimodal transformation module for KAMEL.
Converts tabular data to image (IGTD) and text representations.
"""

import numpy as np
import torch
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from scipy.stats import spearmanr, rankdata
from scipy.spatial.distance import pdist, squareform
import time
import os
import tempfile
import shutil
from transformers import AutoTokenizer


class IGTDTransformer:
    """
    Image Generator for Tabular Data (IGTD) implementation.
    Converts tabular features to 2D image preserving feature topology.
    """
    
    def __init__(
        self, 
        image_size: Tuple[int, int] = (32, 32),
        distance_method: str = 'Pearson',
        pixel_distance_method: str = 'Euclidean',
        max_iterations: int = 1000,
        min_improvement: float = 1e-6,
        random_state: int = 42
    ):
        self.image_size = image_size
        self.distance_method = distance_method  
        self.pixel_distance_method = pixel_distance_method
        self.max_iterations = max_iterations
        self.min_improvement = min_improvement
        self.random_state = random_state
        self.feature_positions = None
        self.pixel_coordinates = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'IGTDTransformer':
        """
        Fit IGTD transformation by optimizing feature-to-pixel mapping.
        
        Args:
            X: Training data (n_samples, n_features)
            
        Returns:
            self
        """
        n_features = X.shape[1]
        img_height, img_width = self.image_size
        
        if n_features > img_height * img_width:
            raise ValueError(
                f"Number of features ({n_features}) exceeds image size "
                f"({img_height} x {img_width} = {img_height * img_width})"
            )
        
        print(f"Fitting IGTD for {n_features} features -> {self.image_size} image")
        
        # Step 1: Compute feature distance ranking matrix
        feature_ranking = self._compute_feature_distance_ranking(X)
        
        # Step 2: Compute pixel distance ranking matrix  
        pixel_coordinates, pixel_ranking = self._compute_pixel_distance_ranking(
            img_height, img_width, n_features
        )
        
        # Step 3: Optimize feature-to-pixel mapping
        optimal_mapping = self._optimize_feature_mapping(
            feature_ranking, pixel_ranking
        )
        
        self.feature_positions = optimal_mapping
        self.pixel_coordinates = pixel_coordinates
        self.is_fitted = True
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform tabular data to image format.
        
        Args:
            X: Input data (n_samples, n_features)
            
        Returns:
            Image data (n_samples, 1, height, width)
        """
        if not self.is_fitted:
            raise ValueError("IGTD transformer must be fitted first")
        
        n_samples, n_features = X.shape
        img_height, img_width = self.image_size
        
        # Normalize features to [0, 1] range
        X_norm = self._normalize_features(X)
        
        # Create image tensor
        images = np.zeros((n_samples, 1, img_height, img_width))
        
        for i in range(n_samples):
            # Map features to pixel positions
            img = np.full((img_height, img_width), 0.0)
            
            for feature_idx in range(n_features):
                row, col = self.pixel_coordinates[self.feature_positions[feature_idx]]
                img[row, col] = X_norm[i, feature_idx]
            
            images[i, 0] = img
        
        return images
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit transformer and transform data."""
        return self.fit(X).transform(X)
    
    def _compute_feature_distance_ranking(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise feature distance ranking matrix."""
        n_features = X.shape[1]
        
        # Add numerical stability: check for constant features
        X_stable = X.copy()
        for i in range(n_features):
            if np.var(X_stable[:, i]) < 1e-8:
                # Add small random noise to constant features
                X_stable[:, i] += np.random.normal(0, 1e-6, X_stable.shape[0])
        
        if self.distance_method == 'Pearson':
            corr = np.corrcoef(X_stable.T)
            # Handle NaN values in correlation matrix
            corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
            distances = 1 - np.abs(corr)
        elif self.distance_method == 'Spearman':
            corr, _ = spearmanr(X_stable)
            corr = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
            distances = 1 - np.abs(corr)
        elif self.distance_method == 'Euclidean':
            distances = squareform(pdist(X_stable.T, metric='euclidean'))
            # Avoid division by zero
            max_dist = np.max(distances)
            if max_dist > 0:
                distances = distances / max_dist
            else:
                distances = np.ones_like(distances) * 0.5
        else:
            raise ValueError(f"Unknown distance method: {self.distance_method}")
        
        # Ensure distances are finite and positive
        distances = np.nan_to_num(distances, nan=0.5, posinf=1.0, neginf=0.0)
        distances = np.clip(distances, 0.0, 1.0)
        
        # Convert to ranking matrix
        tril_indices = np.tril_indices(n_features, k=-1)
        distance_values = distances[tril_indices]
        
        # Handle case where all distances are the same
        if np.all(distance_values == distance_values[0]):
            ranks = np.arange(1, len(distance_values) + 1)
        else:
            ranks = rankdata(distance_values)
        
        ranking = np.zeros((n_features, n_features))
        ranking[tril_indices] = ranks
        ranking = ranking + ranking.T
        
        return ranking
    
    def _compute_pixel_distance_ranking(
        self, 
        img_height: int, 
        img_width: int, 
        n_features: int
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """Compute pixel coordinate distance ranking matrix."""
        
        # Generate pixel coordinates
        coordinates = []
        for row in range(img_height):
            for col in range(img_width):
                coordinates.append((row, col))
                if len(coordinates) >= n_features:
                    break
            if len(coordinates) >= n_features:
                break
        
        coordinates = coordinates[:n_features]
        
        # Compute pairwise pixel distances
        n_pixels = len(coordinates)
        pixel_distances = np.zeros((n_pixels, n_pixels))
        
        for i in range(n_pixels):
            for j in range(n_pixels):
                r1, c1 = coordinates[i]
                r2, c2 = coordinates[j]
                
                if self.pixel_distance_method == 'Euclidean':
                    dist = np.sqrt((r1 - r2)**2 + (c1 - c2)**2)
                elif self.pixel_distance_method == 'Manhattan':
                    dist = abs(r1 - r2) + abs(c1 - c2)
                else:
                    raise ValueError(f"Unknown pixel distance method: {self.pixel_distance_method}")
                
                pixel_distances[i, j] = dist
        
        # Convert to ranking
        tril_indices = np.tril_indices(n_pixels, k=-1)
        ranks = rankdata(pixel_distances[tril_indices])
        ranking = np.zeros((n_pixels, n_pixels))
        ranking[tril_indices] = ranks
        ranking = ranking + ranking.T
        
        return coordinates, ranking
    
    def _optimize_feature_mapping(
        self, 
        feature_ranking: np.ndarray, 
        pixel_ranking: np.ndarray
    ) -> np.ndarray:
        """Optimize feature-to-pixel mapping using iterative swapping."""
        
        np.random.seed(self.random_state)
        n_features = feature_ranking.shape[0]
        
        # Initialize random mapping
        mapping = np.arange(n_features)
        np.random.shuffle(mapping)
        
        # Initial error
        current_error = self._compute_mapping_error(
            feature_ranking, pixel_ranking, mapping
        )
        
        print(f"Initial mapping error: {current_error:.6f}")
        
        # Iterative optimization
        for iteration in range(self.max_iterations):
            improved = False
            
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    # Try swapping positions i and j
                    new_mapping = mapping.copy()
                    new_mapping[i], new_mapping[j] = new_mapping[j], new_mapping[i]
                    
                    new_error = self._compute_mapping_error(
                        feature_ranking, pixel_ranking, new_mapping
                    )
                    
                    if new_error < current_error - self.min_improvement:
                        mapping = new_mapping
                        current_error = new_error
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                print(f"Converged at iteration {iteration}, final error: {current_error:.6f}")
                break
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, error: {current_error:.6f}")
        
        return mapping
    
    def _compute_mapping_error(
        self,
        feature_ranking: np.ndarray,
        pixel_ranking: np.ndarray, 
        mapping: np.ndarray
    ) -> float:
        """Compute error between feature ranking and pixel ranking given mapping."""
        
        # Rearrange feature ranking according to mapping
        reordered_feature_ranking = feature_ranking[mapping][:, mapping]
        
        # Compute absolute difference
        tril_indices = np.tril_indices(len(mapping), k=-1)
        error = np.sum(np.abs(
            reordered_feature_ranking[tril_indices] - pixel_ranking[tril_indices]
        ))
        
        return error
    
    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
        """Normalize features to [0, 1] range."""
        X_norm = X.copy()
        
        for i in range(X.shape[1]):
            feature = X_norm[:, i]
            min_val, max_val = feature.min(), feature.max()
            
            if max_val > min_val:
                X_norm[:, i] = (feature - min_val) / (max_val - min_val)
            else:
                X_norm[:, i] = 0.0
        
        return X_norm


class TextSequentializer:
    """
    Convert tabular data to sequential text representations.
    """
    
    def __init__(
        self,
        tokenizer_name: str = 'distilbert-base-uncased',
        max_length: int = 512,
        template: str = "{feature_name} is {feature_value}"
    ):
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length  
        self.template = template
        self.tokenizer = None
        self.vocab_size = None
        self.is_fitted = False
    
    def fit(self, feature_names: List[str]) -> 'TextSequentializer':
        """
        Fit the text sequentializer.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            self
        """
        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                local_files_only=True,
                cache_dir="../AdvancedMultiModalFusion/models_cache"
            )
        except:
            # Fallback to simple tokenization
            self.tokenizer = SimpleTokenizer()
        
        self.feature_names = feature_names
        self.vocab_size = getattr(self.tokenizer, 'vocab_size', 10000)
        self.is_fitted = True
        
        return self
    
    def transform(
        self, 
        X: np.ndarray, 
        text_sequences: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Transform data to text token sequences.
        
        Args:
            X: Numerical features (n_samples, n_features)
            text_sequences: Pre-generated text sequences (optional)
            
        Returns:
            Dictionary with tokenized text data
        """
        if not self.is_fitted:
            raise ValueError("TextSequentializer must be fitted first")
        
        if text_sequences is None:
            text_sequences = self._generate_text_sequences(X)
        
        # Tokenize sequences
        if hasattr(self.tokenizer, 'batch_encode_plus'):
            # HuggingFace tokenizer
            encoded = self.tokenizer.batch_encode_plus(
                text_sequences,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'text_sequences': text_sequences
            }
        else:
            # Simple tokenizer
            tokens = [self.tokenizer.tokenize(text) for text in text_sequences]
            
            # Pad sequences
            max_len = min(self.max_length, max(len(seq) for seq in tokens) if tokens else 1)
            
            input_ids = []
            attention_masks = []
            
            for token_seq in tokens:
                # Convert to IDs and pad
                ids = [self.tokenizer.token_to_id(token) for token in token_seq[:max_len]]
                attention = [1] * len(ids)
                
                # Pad
                while len(ids) < max_len:
                    ids.append(0)  # PAD token
                    attention.append(0)
                
                input_ids.append(ids)
                attention_masks.append(attention)
            
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
                'text_sequences': text_sequences
            }
    
    def fit_transform(
        self, 
        X: np.ndarray, 
        feature_names: List[str],
        text_sequences: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """Fit and transform in one step."""
        return self.fit(feature_names).transform(X, text_sequences)
    
    def _generate_text_sequences(self, X: np.ndarray) -> List[str]:
        """Generate text sequences from numerical features with robust formatting."""
        sequences = []
        
        # Create a safe template without format specifiers
        safe_template = "{feature_name} is {feature_value}"
        
        for i in range(X.shape[0]):
            feature_texts = []
            for j, feature_name in enumerate(self.feature_names):
                value = X[i, j]
                
                # Always use safe string formatting to avoid format errors
                try:
                    # Convert to appropriate string representation
                    if pd.isna(value) or value is None:
                        formatted_value = "missing"
                    elif isinstance(value, (int, np.integer)):
                        formatted_value = str(int(value))
                    elif isinstance(value, (float, np.floating)):
                        if np.isinf(value):
                            formatted_value = "infinity" if value > 0 else "negative_infinity"
                        elif np.isnan(value):
                            formatted_value = "missing"
                        else:
                            # Use 2 decimal places for float values
                            formatted_value = f"{float(value):.2f}"
                    else:
                        # For any other type, convert to string
                        formatted_value = str(value)
                    
                    # Use safe template for all cases
                    text = safe_template.format(
                        feature_name=feature_name.replace('_', ' '),
                        feature_value=formatted_value
                    )
                    feature_texts.append(text)
                    
                except Exception as e:
                    # Final fallback - create a minimal safe text
                    try:
                        safe_value = str(value) if value is not None else "missing"
                        text = f"{feature_name.replace('_', ' ')} is {safe_value}"
                        feature_texts.append(text)
                    except:
                        # Ultimate fallback
                        text = f"{feature_name.replace('_', ' ')} is unknown"
                        feature_texts.append(text)
            
            sequences.append("; ".join(feature_texts))
        
        return sequences


class SimpleTokenizer:
    """Simple fallback tokenizer when HuggingFace is not available."""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_to_id = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3}
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.next_id = 4
    
    def tokenize(self, text: str) -> List[str]:
        """Simple word tokenization."""
        # Basic preprocessing
        text = text.lower().replace(';', ' ').replace(',', ' ')
        tokens = text.split()
        
        # Add to vocab if needed
        for token in tokens:
            if token not in self.word_to_id and self.next_id < self.vocab_size:
                self.word_to_id[token] = self.next_id
                self.id_to_word[self.next_id] = token
                self.next_id += 1
        
        return tokens
    
    def token_to_id(self, token: str) -> int:
        """Convert token to ID."""
        return self.word_to_id.get(token, 1)  # 1 is [UNK]


class MultiModalTransforms:
    """
    Unified multimodal transformation pipeline.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (32, 32),
        max_text_length: int = 512,
        tokenizer_name: str = 'distilbert-base-uncased'
    ):
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.tokenizer_name = tokenizer_name
        
        self.igtd_transformer = IGTDTransformer(image_size=image_size)
        self.text_sequentializer = TextSequentializer(
            tokenizer_name=tokenizer_name,
            max_length=max_text_length
        )
        
        self.is_fitted = False
    
    def fit(
        self, 
        X_numerical: np.ndarray, 
        feature_names: List[str]
    ) -> 'MultiModalTransforms':
        """
        Fit all transformers.
        
        Args:
            X_numerical: Numerical tabular data for IGTD
            feature_names: Feature names for text generation
            
        Returns:
            self
        """
        print("Fitting IGTD transformer...")
        self.igtd_transformer.fit(X_numerical)
        
        print("Fitting text sequentializer...")
        self.text_sequentializer.fit(feature_names)
        
        self.is_fitted = True
        return self
    
    def transform(
        self, 
        X_numerical: np.ndarray,
        text_sequences: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Transform data to all modalities.
        
        Args:
            X_numerical: Numerical tabular data
            text_sequences: Pre-generated text sequences
            
        Returns:
            Dictionary with tabular, image, and text data
        """
        if not self.is_fitted:
            raise ValueError("MultiModalTransforms must be fitted first")
        
        # Transform to image
        images = self.igtd_transformer.transform(X_numerical)
        
        # Transform to text
        text_data = self.text_sequentializer.transform(X_numerical, text_sequences)
        
        return {
            'tabular': torch.tensor(X_numerical, dtype=torch.float32),
            'image': torch.tensor(images, dtype=torch.float32),
            'text_input_ids': text_data['input_ids'],
            'text_attention_mask': text_data['attention_mask'],
            'text_sequences': text_data['text_sequences']
        }
    
    def fit_transform(
        self,
        X_numerical: np.ndarray,
        feature_names: List[str], 
        text_sequences: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """Fit and transform in one step."""
        return self.fit(X_numerical, feature_names).transform(X_numerical, text_sequences)
