"""
Dynamic Curriculum Resampling and Imbalance Handling for KAMEL.
Implements two-phase training strategy: representation learning -> generalization fine-tuning.
"""

import torch
import torch.utils.data as data
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter
from sklearn.utils import resample
import random


class DynamicCurriculumSampler(data.Sampler):
    
    def __init__(
        self,
        dataset_labels: List[int],
        phase_1_epochs: int = 80,
        total_epochs: int = 200,
        batch_size: int = 256,
        oversampling_ratio: float = 1.0,
        random_state: int = 42
    ):

        self.dataset_labels = np.array(dataset_labels)
        self.phase_1_epochs = phase_1_epochs
        self.total_epochs = total_epochs
        self.batch_size = batch_size
        self.oversampling_ratio = oversampling_ratio
        self.random_state = random_state
        
        # Compute class statistics
        self.class_counts = Counter(dataset_labels)
        self.classes = list(self.class_counts.keys())
        self.num_classes = len(self.classes)
        
        # Find indices for each class
        self.class_indices = {}
        for class_idx in self.classes:
            self.class_indices[class_idx] = np.where(self.dataset_labels == class_idx)[0]
        
        # Identify majority and minority classes
        self.majority_class = max(self.class_counts, key=self.class_counts.get)
        self.minority_class = min(self.class_counts, key=self.class_counts.get)
        self.max_count = self.class_counts[self.majority_class]
        self.min_count = self.class_counts[self.minority_class]
        
        print(f"Class distribution: {dict(self.class_counts)}")
        print(f"Imbalance ratio: {self.min_count / self.max_count:.4f}")
        
        self.current_epoch = 0
        np.random.seed(random_state)
        random.seed(random_state)
    
    def set_epoch(self, epoch: int):

        self.current_epoch = epoch
    
    def _get_sampling_strategy(self) -> str:

        if self.current_epoch < self.phase_1_epochs:
            return "balanced_oversampling"
        else:
            return "gradual_annealing"
    
    def _balanced_oversampling(self) -> List[int]:

        # Target count for balanced sampling
        target_count = int(self.max_count * self.oversampling_ratio)
        
        sampled_indices = []
        
        for class_idx in self.classes:
            class_indices = self.class_indices[class_idx]
            class_count = len(class_indices)
            
            if class_count < target_count:
                # Oversample minority class
                oversampled = resample(
                    class_indices,
                    n_samples=target_count,
                    replace=True,
                    random_state=self.random_state + self.current_epoch
                )
                sampled_indices.extend(oversampled)
            else:
                # Undersample majority class
                undersampled = resample(
                    class_indices,
                    n_samples=target_count,
                    replace=False,
                    random_state=self.random_state + self.current_epoch
                )
                sampled_indices.extend(undersampled)
        
        # Shuffle the combined indices
        np.random.shuffle(sampled_indices)
        return sampled_indices
    
    def _gradual_annealing(self) -> List[int]:

        # Calculate annealing progress (0 to 1)
        phase_2_progress = (self.current_epoch - self.phase_1_epochs) / (self.total_epochs - self.phase_1_epochs)
        phase_2_progress = min(phase_2_progress, 1.0)
        
        sampled_indices = []
        
        for class_idx in self.classes:
            class_indices = self.class_indices[class_idx]
            original_count = len(class_indices)
            
            # Interpolate between balanced and original count
            balanced_count = int(self.max_count * self.oversampling_ratio)
            target_count = int(balanced_count + phase_2_progress * (original_count - balanced_count))
            target_count = max(target_count, original_count // 2)  # Minimum threshold
            
            if target_count > original_count:
                # Still need oversampling
                oversampled = resample(
                    class_indices,
                    n_samples=target_count,
                    replace=True,
                    random_state=self.random_state + self.current_epoch
                )
                sampled_indices.extend(oversampled)
            else:
                # Use subset of original samples
                subset = resample(
                    class_indices,
                    n_samples=target_count,
                    replace=False,
                    random_state=self.random_state + self.current_epoch
                )
                sampled_indices.extend(subset)
        
        # Shuffle the combined indices
        np.random.shuffle(sampled_indices)
        return sampled_indices
    
    def __iter__(self):

        strategy = self._get_sampling_strategy()
        
        if strategy == "balanced_oversampling":
            indices = self._balanced_oversampling()
        elif strategy == "gradual_annealing":
            indices = self._gradual_annealing()
        else:
            # Fallback to original sampling
            indices = list(range(len(self.dataset_labels)))
            np.random.shuffle(indices)
        
        return iter(indices)
    
    def __len__(self):
  
        strategy = self._get_sampling_strategy()
        
        if strategy == "balanced_oversampling":
            return int(self.max_count * self.oversampling_ratio * self.num_classes)
        elif strategy == "gradual_annealing":
            # Estimate length based on annealing progress
            phase_2_progress = (self.current_epoch - self.phase_1_epochs) / (self.total_epochs - self.phase_1_epochs)
            phase_2_progress = min(phase_2_progress, 1.0)
            
            balanced_length = int(self.max_count * self.oversampling_ratio * self.num_classes)
            original_length = len(self.dataset_labels)
            
            return int(balanced_length + phase_2_progress * (original_length - balanced_length))
        else:
            return len(self.dataset_labels)


class SMOTESampler(data.Sampler):

    
    def __init__(
        self,
        dataset_features: np.ndarray,
        dataset_labels: List[int],
        k_neighbors: int = 5,
        sampling_ratio: float = 1.0,
        random_state: int = 42
    ):

        self.dataset_features = dataset_features
        self.dataset_labels = np.array(dataset_labels)
        self.k_neighbors = k_neighbors
        self.sampling_ratio = sampling_ratio
        self.random_state = random_state
        
        # Compute class statistics
        self.class_counts = Counter(dataset_labels)
        self.classes = list(self.class_counts.keys())
        self.majority_class = max(self.class_counts, key=self.class_counts.get)
        self.max_count = self.class_counts[self.majority_class]
        
        np.random.seed(random_state)
        
    def _generate_smote_samples(self, class_idx: int, target_samples: int) -> Tuple[np.ndarray, np.ndarray]:

        # Get minority class samples
        minority_mask = (self.dataset_labels == class_idx)
        minority_features = self.dataset_features[minority_mask]
        
        if len(minority_features) < self.k_neighbors:
            # Not enough samples for SMOTE, just replicate
            indices = np.random.choice(len(minority_features), target_samples, replace=True)
            return minority_features[indices], np.full(target_samples, class_idx)
        
        synthetic_features = []
        
        for _ in range(target_samples):
            # Select random minority sample
            idx = np.random.randint(0, len(minority_features))
            sample = minority_features[idx]
            
            # Find k nearest neighbors
            distances = np.sum((minority_features - sample) ** 2, axis=1)
            neighbor_indices = np.argsort(distances)[1:self.k_neighbors + 1]  # Exclude self
            
            # Select random neighbor
            neighbor_idx = np.random.choice(neighbor_indices)
            neighbor = minority_features[neighbor_idx]
            
            # Generate synthetic sample
            alpha = np.random.random()
            synthetic_sample = sample + alpha * (neighbor - sample)
            synthetic_features.append(synthetic_sample)
        
        synthetic_features = np.array(synthetic_features)
        synthetic_labels = np.full(target_samples, class_idx)
        
        return synthetic_features, synthetic_labels
    
    def get_balanced_dataset(self) -> Tuple[np.ndarray, np.ndarray]:

        all_features = [self.dataset_features]
        all_labels = [self.dataset_labels]
        
        for class_idx in self.classes:
            current_count = self.class_counts[class_idx]
            target_count = int(self.max_count * self.sampling_ratio)
            
            if current_count < target_count:
                # Generate synthetic samples
                num_synthetic = target_count - current_count
                synthetic_features, synthetic_labels = self._generate_smote_samples(
                    class_idx, num_synthetic
                )
                
                all_features.append(synthetic_features)
                all_labels.append(synthetic_labels)
        
        # Combine all features and labels
        balanced_features = np.vstack(all_features)
        balanced_labels = np.hstack(all_labels)
        
        # Shuffle
        indices = np.random.permutation(len(balanced_labels))
        return balanced_features[indices], balanced_labels[indices]
    
    def __iter__(self):
 
        # Generate SMOTE balanced dataset
        balanced_features, balanced_labels = self.get_balanced_dataset()
        # Return indices for the balanced dataset
        return iter(range(len(balanced_labels)))
    
    def __len__(self):
  
        # Calculate total samples after SMOTE
        total_samples = len(self.dataset_labels)
        for class_idx in self.classes:
            current_count = self.class_counts[class_idx]
            target_count = int(self.max_count * self.sampling_ratio)
            if current_count < target_count:
                total_samples += (target_count - current_count)
        return total_samples


class ImbalanceAwareTrainer:

    
    def __init__(
        self,
        sampling_strategy: str = "curriculum",
        loss_strategy: str = "ldam",
        phase_1_epochs: int = 80,
        total_epochs: int = 200,
        **kwargs
    ):

        self.sampling_strategy = sampling_strategy
        self.loss_strategy = loss_strategy
        self.phase_1_epochs = phase_1_epochs
        self.total_epochs = total_epochs
        self.kwargs = kwargs
        
        self.current_epoch = 0
        self.samplers = {}
    
    def create_sampler(
        self,
        dataset_labels: List[int],
        dataset_features: Optional[np.ndarray] = None,
        **kwargs
    ) -> data.Sampler:

        if self.sampling_strategy == "curriculum":
            return DynamicCurriculumSampler(
                dataset_labels=dataset_labels,
                phase_1_epochs=self.phase_1_epochs,
                total_epochs=self.total_epochs,
                **kwargs
            )
        elif self.sampling_strategy == "smote":
            if dataset_features is None:
                raise ValueError("SMOTE sampling requires dataset features")
            # Filter kwargs for SMOTE sampler (doesn't accept batch_size)
            smote_kwargs = {k: v for k, v in kwargs.items() if k != 'batch_size'}
            return SMOTESampler(
                dataset_features=dataset_features,
                dataset_labels=dataset_labels,
                **smote_kwargs
            )
        elif self.sampling_strategy == "balanced":
            # Balanced sampling without oversampling - use WeightedRandomSampler
            from collections import Counter
            class_counts = Counter(dataset_labels)
            weights = []
            for label in dataset_labels:
                weights.append(1.0 / class_counts[label])
            return data.WeightedRandomSampler(weights, len(dataset_labels), replacement=False)
        else:
            # Default to random sampling
            return data.RandomSampler(range(len(dataset_labels)))
    
    def update_epoch(self, epoch: int, samplers: Optional[List[data.Sampler]] = None):

        self.current_epoch = epoch
        
        if samplers:
            for sampler in samplers:
                if hasattr(sampler, 'set_epoch'):
                    sampler.set_epoch(epoch)
    
    def get_current_phase(self) -> str:
        """Get current training phase."""
        if self.current_epoch < self.phase_1_epochs:
            return "representation_learning"
        else:
            return "generalization_finetuning"

