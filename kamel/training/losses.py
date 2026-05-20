"""
Loss functions for imbalanced tabular classification.
Implements LDAM, Focal Loss, and other imbalance-aware losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Union


class LDAMLoss(nn.Module):

    
    def __init__(
        self,
        cls_num_list: List[int],
        max_margin: float = 0.5,
        scale_factor: float = 30.0,
        weight: Optional[torch.Tensor] = None
    ):

        super(LDAMLoss, self).__init__()
        
        # Compute margin list based on class frequencies
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_margin / np.max(m_list))
        
        self.register_buffer('m_list', torch.tensor(m_list, dtype=torch.float32))
        self.scale_factor = scale_factor
        self.weight = weight
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        batch_size = logits.shape[0]
        num_classes = logits.shape[1]
        
        # Create index matrix
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, targets.data.view(-1, 1), True)
        
        # Apply margin to correct class logits
        index_float = index.float()
        # Ensure m_list is on the same device as logits
        m_list_device = self.m_list.to(logits.device)
        batch_m = torch.matmul(m_list_device[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        
        # Subtract margin from correct class logits
        logits_m = logits - batch_m
        
        # Apply margin only to correct classes
        output = torch.where(index, logits_m, logits)
        
        # Apply scaling and compute cross entropy
        # Ensure weight is on the same device as logits
        weight = self.weight.to(logits.device) if self.weight is not None else None
        return F.cross_entropy(self.scale_factor * output, targets, weight=weight)


class FocalLoss(nn.Module):

    
    def __init__(
        self,
        alpha: Optional[Union[float, torch.Tensor]] = None,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):

        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        # Standard cross entropy
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        
        # Compute p_t
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on effective number of samples.
    
    Reference: Class-Balanced Loss Based on Effective Number of Samples (CVPR 2019)
    """
    
    def __init__(
        self,
        cls_num_list: List[int],
        beta: float = 0.9999,
        loss_type: str = 'focal',
        gamma: float = 2.0
    ):
        """
        Initialize Class-Balanced Loss.
        
        Args:
            cls_num_list: List of sample counts for each class
            beta: Hyperparameter for re-weighting
            loss_type: Base loss type ('focal', 'sigmoid', 'softmax')
            gamma: Gamma parameter for focal loss
        """
        super(ClassBalancedLoss, self).__init__()
        
        # Compute effective numbers
        effective_num = 1.0 - np.power(beta, cls_num_list)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(cls_num_list)
        
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        self.loss_type = loss_type
        self.gamma = gamma
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Class-Balanced loss.
        
        Args:
            logits: Model predictions (batch_size, num_classes)
            targets: True labels (batch_size,)
            
        Returns:
            Class-balanced loss
        """
        # Ensure weights are on the same device as logits
        weights = self.weights.to(logits.device)
        
        if self.loss_type == 'focal':
            cb_loss = FocalLoss(weight=weights, gamma=self.gamma)
            return cb_loss(logits, targets)
        elif self.loss_type == 'softmax':
            return F.cross_entropy(logits, targets, weight=weights)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class ImbalanceAwareLoss(nn.Module):
    """
    Combined loss function that adaptively uses different losses based on training phase.
    """
    
    def __init__(
        self,
        cls_num_list: List[int],
        loss_type: str = 'ldam',
        phase_1_epochs: int = 160,
        phase_2_epochs: int = 180,
        **kwargs
    ):
        """
        Initialize Imbalance-Aware Loss.
        
        Args:
            cls_num_list: List of sample counts for each class
            loss_type: Primary loss type ('ldam', 'focal', 'cb')
            phase_1_epochs: Number of epochs for phase 1 (representation learning)
            phase_2_epochs: Total epochs (phase 2 starts after phase_1_epochs)
            **kwargs: Additional arguments for specific losses
        """
        super(ImbalanceAwareLoss, self).__init__()
        
        self.cls_num_list = cls_num_list
        self.loss_type = loss_type
        self.phase_1_epochs = phase_1_epochs
        self.phase_2_epochs = phase_2_epochs
        self.current_epoch = 0
        
        # Initialize primary loss
        if loss_type == 'ldam':
            self.primary_loss = LDAMLoss(cls_num_list, **kwargs)
        elif loss_type == 'focal':
            self.primary_loss = FocalLoss(**kwargs)
        elif loss_type == 'cb':
            self.primary_loss = ClassBalancedLoss(cls_num_list, **kwargs)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Backup cross-entropy loss
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive loss based on current training phase.
        
        Args:
            logits: Model predictions (batch_size, num_classes)
            targets: True labels (batch_size,)
            
        Returns:
            Adaptive loss
        """
        if self.current_epoch < self.phase_1_epochs:
            # Phase 1: Focus on representation learning with primary loss
            return self.primary_loss(logits, targets)
        else:
            # Phase 2: Combine primary loss with cross-entropy for fine-tuning
            primary = self.primary_loss(logits, targets)
            ce = self.ce_loss(logits, targets)
            
            # Weighted combination (more weight on CE in later phases)
            weight = (self.current_epoch - self.phase_1_epochs) / (self.phase_2_epochs - self.phase_1_epochs)
            weight = min(weight, 1.0)
            
            return (1 - weight * 0.5) * primary + weight * 0.5 * ce
    
    def update_epoch(self, epoch: int):
        """Update current epoch for adaptive loss computation."""
        self.current_epoch = epoch


class DistributionAwareLoss(nn.Module):
    """
    Distribution-aware loss that adapts to the local class distribution in each batch.
    """
    
    def __init__(
        self,
        global_cls_num_list: List[int],
        temperature: float = 1.0,
        smooth_factor: float = 0.1
    ):
        """
        Initialize Distribution-Aware Loss.
        
        Args:
            global_cls_num_list: Global class distribution
            temperature: Temperature for softmax
            smooth_factor: Smoothing factor for batch distribution
        """
        super(DistributionAwareLoss, self).__init__()
        
        total_samples = sum(global_cls_num_list)
        global_probs = torch.tensor([count / total_samples for count in global_cls_num_list])
        self.register_buffer('global_probs', global_probs)
        
        self.temperature = temperature
        self.smooth_factor = smooth_factor
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute distribution-aware loss.
        
        Args:
            logits: Model predictions (batch_size, num_classes)
            targets: True labels (batch_size,)
            
        Returns:
            Distribution-aware loss
        """
        batch_size = logits.shape[0]
        num_classes = logits.shape[1]
        
        # Compute batch class distribution
        batch_counts = torch.bincount(targets, minlength=num_classes).float()
        batch_probs = batch_counts / batch_size
        
        # Smooth with global distribution
        adapted_probs = (1 - self.smooth_factor) * batch_probs + self.smooth_factor * self.global_probs
        
        # Compute adaptive weights (inverse of adapted probabilities)
        weights = 1.0 / (adapted_probs + 1e-8)
        weights = weights / weights.sum() * num_classes
        
        # Apply weights to cross entropy
        return F.cross_entropy(logits / self.temperature, targets, weight=weights)


def get_loss_function(
    loss_type: str,
    cls_num_list: List[int],
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: Type of loss ('ldam', 'focal', 'cb', 'adaptive')
        cls_num_list: Class sample counts
        **kwargs: Additional arguments
        
    Returns:
        Loss function instance
    """
    if loss_type == 'ldam':
        # Filter out parameters that are not for LDAM loss
        ldam_kwargs = {}
        valid_ldam_params = {'max_margin', 'scale_factor', 'weight'}
        for key, value in kwargs.items():
            if key in valid_ldam_params:
                ldam_kwargs[key] = value
        
        # Convert weight list to tensor if provided
        if 'weight' in ldam_kwargs and isinstance(ldam_kwargs['weight'], list):
            ldam_kwargs['weight'] = torch.tensor(ldam_kwargs['weight'], dtype=torch.float32)
        return LDAMLoss(cls_num_list, **ldam_kwargs)
    elif loss_type == 'focal':
        # Filter out parameters that are not for Focal loss
        focal_kwargs = {}
        valid_focal_params = {'alpha', 'gamma', 'weight', 'reduction'}
        for key, value in kwargs.items():
            if key in valid_focal_params:
                focal_kwargs[key] = value
        return FocalLoss(**focal_kwargs)
    elif loss_type == 'cb':
        # Filter out parameters that are not for Class Balanced loss
        cb_kwargs = {}
        valid_cb_params = {'beta', 'loss_type', 'gamma'}
        for key, value in kwargs.items():
            if key in valid_cb_params:
                cb_kwargs[key] = value
        return ClassBalancedLoss(cls_num_list, **cb_kwargs)
    elif loss_type == 'adaptive':
        return ImbalanceAwareLoss(cls_num_list, **kwargs)
    elif loss_type == 'distribution_aware':
        return DistributionAwareLoss(cls_num_list, **kwargs)
    elif loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
