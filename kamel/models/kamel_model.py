#!/usr/bin/env python3
"""
Main KAMEL model implementation.
Integrates KAN encoders, MoE fusion, and imbalance-aware training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union

from .kan_encoders import UnifiedKANEncoders
from .moe_fusion import AdaptiveMoEFusion, HierarchicalMoEFusion
from ..training.losses import get_loss_function
from ..data.transforms import MultiModalTransforms


class KAMELModel(nn.Module):
    """
    KAMEL: KAN-fused Adaptive Mixture-of-Experts for Tabular Learning
    
    Architecture:
    1. Multimodal KAN Encoders (Tabular, Image, Text)
    2. Adaptive MoE Fusion
    3. Imbalance-Aware Classification Head
    """
    
    def __init__(
        self,
        # Data configuration
        tabular_dim: int,
        num_classes: int = 2,
        vocab_size: int = 10000,
        
        # Architecture configuration
        embed_dim: int = 64,
        image_size: Tuple[int, int] = (32, 32),
        max_text_length: int = 512,
        
        # KAN encoder configuration
        kan_type: str = 'fast_kan',
        kan_hidden_dims: List[int] = [128],
        encoder_dropout: float = 0.1,
        
        # MoE fusion configuration
        num_unimodal_experts: int = 3,
        num_crossmodal_experts: int = 3,
        num_synergy_experts: int = 1,
        use_hierarchical_moe: bool = False,
        moe_dropout: float = 0.1,
        use_sparse_routing: bool = True,
        moe_top_k: int = 2,
        gate_temperature: float = 1.0,
        
        # Classification configuration
        classifier_hidden_dims: List[int] = [32],
        classifier_dropout: float = 0.5,
        
        # Training configuration
        device: str = 'cuda'
    ):
        """Initialize KAMEL model."""
        super(KAMELModel, self).__init__()
        
        self.tabular_dim = tabular_dim
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.kan_type = kan_type
        self.device = device
        
        # KAN Encoders
        self.encoders = UnifiedKANEncoders(
            tabular_config={
                'input_dim': tabular_dim,
                'hidden_dims': kan_hidden_dims,
                'kan_type': kan_type,
                'dropout': encoder_dropout
            },
            image_config={
                'input_shape': (1, *image_size),
                'kan_hidden_dims': kan_hidden_dims,
                'kan_type': kan_type,
                'dropout': encoder_dropout
            },
            text_config={
                'vocab_size': vocab_size,
                'max_seq_length': max_text_length,
                'kan_hidden_dims': kan_hidden_dims,
                'kan_type': kan_type,
                'dropout': encoder_dropout
            },
            embed_dim=embed_dim,
            device=device
        )
        
        # MoE Fusion
        if use_hierarchical_moe:
            self.fusion = HierarchicalMoEFusion(
                embed_dim=embed_dim,
                num_experts_per_type=2,
                dropout=moe_dropout,
                device=device
            )
        else:
            self.fusion = AdaptiveMoEFusion(
                embed_dim=embed_dim,
                num_unimodal_experts=num_unimodal_experts,
                num_crossmodal_experts=num_crossmodal_experts,
                num_synergy_experts=num_synergy_experts,
                dropout=moe_dropout,
                use_sparse_routing=use_sparse_routing,
                top_k=moe_top_k,
                gate_temperature=gate_temperature,
                device=device
            )
        
        # Classification Head
        classifier_layers = []
        in_dim = embed_dim
        
        for hidden_dim in classifier_hidden_dims:
            classifier_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(classifier_dropout)
            ])
            in_dim = hidden_dim
        
        classifier_layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
        
        # Move to device
        self.to(device)
    
    def forward(
        self,
        tabular_data: torch.Tensor,
        image_data: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        return_auxiliary: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Forward pass through KAMEL model."""
        # Encode each modality
        tabular_embed, image_embed, text_embed = self.encoders(
            tabular_data=tabular_data,
            image_data=image_data,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask
        )
        
        # Fusion through MoE
        fused_features, moe_auxiliary = self.fusion(
            tabular_embed=tabular_embed,
            image_embed=image_embed,
            text_embed=text_embed,
            training=self.training
        )
        
        # Classification
        logits = self.classifier(fused_features)
        
        if return_auxiliary:
            auxiliary = {
                'tabular_embed': tabular_embed,
                'image_embed': image_embed,
                'text_embed': text_embed,
                'fused_features': fused_features,
                'moe_auxiliary': moe_auxiliary
            }
            return logits, auxiliary
        
        return logits
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss_fn,
        auxiliary: Optional[Dict[str, Any]] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute total loss including MoE regularization."""
        losses = {}
        
        # Main classification loss
        main_loss = loss_fn(logits, targets)
        losses['main_loss'] = main_loss
        
        # MoE regularization
        if auxiliary and 'moe_auxiliary' in auxiliary:
            moe_aux = auxiliary['moe_auxiliary']
            
            if 'load_balancing_loss' in moe_aux:
                load_loss = moe_aux['load_balancing_loss']
                losses['load_balancing_loss'] = load_loss
            
            if 'diversity_loss' in moe_aux:
                diversity_loss = moe_aux['diversity_loss']
                losses['diversity_loss'] = diversity_loss
        
        # Total loss
        total_loss = main_loss
        for key, loss in losses.items():
            if key != 'main_loss':
                total_loss = total_loss + 0.01 * loss
        
        losses['total_loss'] = total_loss
        return losses
    
    def freeze_encoders(self):
        """Freeze encoder parameters for Stage 2 training (fusion only)."""
        for param in self.encoders.parameters():
            param.requires_grad = False
        print("[OK] Encoders frozen for fusion-only training")
    
    def unfreeze_encoders(self):
        """Unfreeze encoder parameters for end-to-end training.""" 
        for param in self.encoders.parameters():
            param.requires_grad = True
        print("[OK] Encoders unfrozen for end-to-end training")
    
    def load_encoder_weights(self, encoder_state_dict: Dict[str, torch.Tensor]):
        """Load pre-trained encoder weights."""
        # Filter encoder parameters
        encoder_keys = [key for key in encoder_state_dict.keys() 
                       if any(encoder_name in key for encoder_name in ['tabular_encoder', 'image_encoder', 'text_encoder'])]
        
        # Load weights
        current_state = self.state_dict()
        for key in encoder_keys:
            if key in current_state:
                current_state[key] = encoder_state_dict[key]
        
        self.load_state_dict(current_state)
        print(f"[OK] Loaded {len(encoder_keys)} encoder parameters")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        encoder_params = sum(p.numel() for p in self.encoders.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'encoder_parameters': encoder_params,
            'fusion_parameters': fusion_params,
            'classifier_parameters': classifier_params,
            'embed_dim': self.embed_dim,
            'num_classes': self.num_classes,
            'device': str(next(self.parameters()).device)
        }
