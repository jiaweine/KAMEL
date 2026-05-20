"""
Adaptive Mixture-of-Experts (MoE) Fusion module for KAMEL.
Implements heterogeneous MLP experts with intelligent gating for multimodal fusion.
Optimized for speed and inspired by I2MoE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class GatingNetwork(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        hidden_dim: int = 128,
        temperature: float = 1.0,
        use_noise: bool = True,
        noise_std: float = 1.0,
        use_sparse_routing: bool = True,
        top_k: int = 2,
        device: str = 'cuda'
    ):

        super(GatingNetwork, self).__init__()
        
        self.num_experts = num_experts
        self.temperature = temperature
        self.use_noise = use_noise
        self.noise_std = noise_std
        self.use_sparse_routing = use_sparse_routing
        self.top_k = min(top_k, num_experts)
        self.device = device
        
        # Gating network layers - reduced dropout for better routing
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Reduced dropout
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),  # Reduced dropout
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
        # Initialize with small weights
        for module in self.gate.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.1)
                nn.init.constant_(module.bias, 0)
        
        # Ensure all layers are on the correct device
        self.to(device)
    
    def forward(self, x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:

        # Ensure input is on the correct device
        x = x.to(self.device)
        
        # Compute gate logits
        gate_logits = self.gate(x)
        
        # Add noise during training for exploration
        if training and self.use_noise:
            noise = torch.randn_like(gate_logits) * self.noise_std
            gate_logits = gate_logits + noise
        
        if self.use_sparse_routing:
            # Sparse top-k routing
            top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=1)
            
            # Create sparse gate weights
            gate_weights = torch.zeros_like(gate_logits)
            gate_weights.scatter_(1, top_k_indices, F.softmax(top_k_logits / self.temperature, dim=1))
        else:
            # Dense routing (original behavior)
            gate_weights = F.softmax(gate_logits / self.temperature, dim=1)
        
        return gate_weights, gate_logits
    
    def load_balancing_loss(self, gate_weights: torch.Tensor) -> torch.Tensor:

        # Compute expert utilization rates
        expert_rates = torch.mean(gate_weights, dim=0)  # (num_experts,)
        
        # Compute coefficient of variation (std/mean) for load balancing
        # Lower CV means more balanced utilization
        mean_rate = torch.mean(expert_rates)
        std_rate = torch.std(expert_rates)
        cv_loss = std_rate / (mean_rate + 1e-8)
        
        return cv_loss
    
    def diversity_loss(self, gate_weights: torch.Tensor) -> torch.Tensor:

        # Compute pairwise cosine similarity between expert usage patterns
        # Normalize gate weights
        gate_norm = F.normalize(gate_weights.T, p=2, dim=1)  # (num_experts, batch_size)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(gate_norm, gate_norm.T)  # (num_experts, num_experts)
        
        # Remove diagonal (self-similarity)
        mask = torch.eye(self.num_experts, device=self.device)
        off_diagonal = similarity_matrix * (1 - mask)
        
        # Penalize high similarity (encourage diversity)
        diversity_loss = torch.mean(torch.abs(off_diagonal))
        
        return diversity_loss


class ExpertNetwork(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        expert_type: str = 'mlp_relu',
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        device: str = 'cuda'
    ):
        super(ExpertNetwork, self).__init__()
        
        self.expert_type = expert_type
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Create heterogeneous MLP architectures
        layers = []
        in_dim = input_dim
        
        # Define activation functions for different expert types
        if expert_type == 'mlp_relu':
            activation = nn.ReLU()
            norm_layer = nn.BatchNorm1d
        elif expert_type == 'mlp_gelu':
            activation = nn.GELU()
            norm_layer = nn.LayerNorm
        elif expert_type == 'mlp_swish':
            activation = nn.SiLU()  # Swish activation
            norm_layer = nn.BatchNorm1d
        elif expert_type == 'mlp_leaky':
            activation = nn.LeakyReLU(0.1)
            norm_layer = nn.LayerNorm
        elif expert_type == 'mlp_elu':
            activation = nn.ELU()
            norm_layer = nn.BatchNorm1d
        elif expert_type == 'mlp_prelu':
            activation = nn.PReLU()
            norm_layer = nn.LayerNorm
        elif expert_type == 'mlp_mish':
            activation = nn.Mish()
            norm_layer = nn.BatchNorm1d
        else:
            # Default to ReLU
            activation = nn.ReLU()
            norm_layer = nn.BatchNorm1d
        
        # Build layers with heterogeneous designs
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            
            # Vary normalization placement for heterogeneity
            if expert_type in ['mlp_gelu', 'mlp_leaky', 'mlp_prelu']:
                # Pre-activation normalization
                if norm_layer == nn.LayerNorm:
                    layers.append(norm_layer(hidden_dim))
                else:
                    layers.append(norm_layer(hidden_dim))
                layers.append(activation)
            else:
                # Post-activation normalization
                layers.append(activation)
                if norm_layer == nn.LayerNorm:
                    layers.append(norm_layer(hidden_dim))
                else:
                    layers.append(norm_layer(hidden_dim))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        # Ensure all layers are on the correct device
        self.device = device
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Ensure input is on the correct device
        x = x.to(self.device)
        return self.network(x)


class AdaptiveMoEFusion(nn.Module):

    
    def __init__(
        self,
        embed_dim: int = 64,
        num_unimodal_experts: int = 3,
        num_crossmodal_experts: int = 3, 
        num_synergy_experts: int = 1,
        expert_hidden_dim: int = 128,
        expert_num_layers: int = 2,
        gate_hidden_dim: int = 128,
        gate_temperature: float = 1.0,
        dropout: float = 0.1,
        use_sparse_routing: bool = True,
        top_k: int = 2,
        load_balance_weight: float = 0.01,
        diversity_weight: float = 0.01,
        device: str = 'cuda'
    ):

        super(AdaptiveMoEFusion, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_unimodal_experts = num_unimodal_experts
        self.num_crossmodal_experts = num_crossmodal_experts
        self.num_synergy_experts = num_synergy_experts
        self.use_sparse_routing = use_sparse_routing
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight
        self.diversity_weight = diversity_weight
        self.device = device
        
        # Total number of experts
        self.total_experts = num_unimodal_experts + num_crossmodal_experts + num_synergy_experts
        
        # Input dimension is 3 * embed_dim (concatenated modalities)
        input_dim = 3 * embed_dim
        
        # Gating network
        self.gating_network = GatingNetwork(
            input_dim=input_dim,
            num_experts=self.total_experts,
            hidden_dim=gate_hidden_dim,
            temperature=gate_temperature,
            use_sparse_routing=use_sparse_routing,
            top_k=top_k,
            device=device
        )
        
        # Expert networks with heterogeneous MLP architectures for diversity
        self.experts = nn.ModuleList()
        
        # Unimodal experts: focus on individual modalities with different activations
        unimodal_types = ['mlp_relu', 'mlp_gelu', 'mlp_swish']
        unimodal_hidden_dims = [expert_hidden_dim, expert_hidden_dim + 32, expert_hidden_dim - 16]
        for i in range(num_unimodal_experts):
            expert_type = unimodal_types[i % len(unimodal_types)]
            hidden_dim = unimodal_hidden_dims[i % len(unimodal_hidden_dims)]
            expert = ExpertNetwork(
                input_dim=input_dim,
                output_dim=embed_dim,
                expert_type=expert_type,
                hidden_dim=max(hidden_dim, 32),  # Ensure min hidden dim
                num_layers=expert_num_layers,
                dropout=dropout,
                device=device
            )
            self.experts.append(expert)
        
        # Cross-modal experts: focus on pairwise interactions with varied architectures
        crossmodal_types = ['mlp_leaky', 'mlp_elu', 'mlp_prelu']
        crossmodal_hidden_dims = [expert_hidden_dim + 16, expert_hidden_dim, expert_hidden_dim + 48]
        for i in range(num_crossmodal_experts):
            expert_type = crossmodal_types[i % len(crossmodal_types)]
            hidden_dim = crossmodal_hidden_dims[i % len(crossmodal_hidden_dims)]
            expert = ExpertNetwork(
                input_dim=input_dim,
                output_dim=embed_dim,
                expert_type=expert_type,
                hidden_dim=max(hidden_dim, 32),  # Ensure min hidden dim
                num_layers=expert_num_layers,
                dropout=dropout,
                device=device
            )
            self.experts.append(expert)
        
        # Synergy experts: focus on three-way interactions with larger capacity
        for i in range(num_synergy_experts):
            expert = ExpertNetwork(
                input_dim=input_dim,
                output_dim=embed_dim,
                expert_type='mlp_mish',  # Use Mish activation for synergy
                hidden_dim=expert_hidden_dim * 2,  # Larger capacity for complex interactions
                num_layers=expert_num_layers + 1,  # Deeper for synergy modeling
                dropout=dropout * 0.8,  # Lower dropout for synergy expert
                device=device
            )
            self.experts.append(expert)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)
        
        # Ensure all components are on the correct device
        self.to(device)
    
    def forward(
        self,
        tabular_embed: torch.Tensor,
        image_embed: torch.Tensor,
        text_embed: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        # Ensure all inputs are on the correct device
        tabular_embed = tabular_embed.to(self.device)
        image_embed = image_embed.to(self.device)
        text_embed = text_embed.to(self.device)
        
        batch_size = tabular_embed.shape[0]
        
        # Concatenate all modalities
        combined_input = torch.cat([tabular_embed, image_embed, text_embed], dim=1)
        
        # Gating network
        gate_weights, gate_logits = self.gating_network(combined_input, training)
        
        # Expert outputs
        expert_outputs = []
        for expert in self.experts:
            output = expert(combined_input)
            expert_outputs.append(output)
        
        # Stack expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch_size, num_experts, embed_dim)
        
        # Weighted combination
        gate_weights = gate_weights.unsqueeze(-1)  # (batch_size, num_experts, 1)
        fused_output = torch.sum(expert_outputs * gate_weights, dim=1)  # (batch_size, embed_dim)
        
        # Output projection
        fused_output = self.output_proj(fused_output)
        fused_output = self.output_norm(fused_output)
        
        # Auxiliary outputs for analysis
        gate_weights_2d = gate_weights.squeeze(-1)
        auxiliary = {
            'gate_weights': gate_weights_2d,
            'gate_logits': gate_logits,
            'expert_outputs': expert_outputs,
            'load_balancing_loss': self.gating_network.load_balancing_loss(gate_weights_2d),
            'diversity_loss': self.gating_network.diversity_loss(gate_weights_2d)
        }
        
        return fused_output, auxiliary
    
    def get_expert_utilization(self, gate_weights: torch.Tensor) -> torch.Tensor:

        return torch.mean(gate_weights, dim=0)
    
    def get_fusion_loss(self, auxiliary: Dict[str, torch.Tensor]) -> torch.Tensor:

        load_balance_loss = auxiliary['load_balancing_loss']
        diversity_loss = auxiliary['diversity_loss']
        
        # Combine regularization terms
        total_loss = (self.load_balance_weight * load_balance_loss + 
                     self.diversity_weight * diversity_loss)
        
        return total_loss


class HierarchicalMoEFusion(nn.Module):

    
    def __init__(
        self,
        embed_dim: int = 64,
        num_experts_per_type: int = 2,
        device: str = 'cuda'
    ):

        super(HierarchicalMoEFusion, self).__init__()
        
        self.embed_dim = embed_dim
        self.device = device
        
        # Type-specific MoE modules
        self.unimodal_moe = AdaptiveMoEFusion(
            embed_dim=embed_dim,
            num_unimodal_experts=num_experts_per_type,
            num_crossmodal_experts=0,
            num_synergy_experts=0,
            device=device
        )
        
        self.crossmodal_moe = AdaptiveMoEFusion(
            embed_dim=embed_dim,
            num_unimodal_experts=0,
            num_crossmodal_experts=num_experts_per_type,
            num_synergy_experts=0,
            device=device
        )
        
        self.synergy_moe = AdaptiveMoEFusion(
            embed_dim=embed_dim,
            num_unimodal_experts=0,
            num_crossmodal_experts=0,
            num_synergy_experts=num_experts_per_type,
            device=device
        )
        
        # Final fusion
        self.final_fusion = nn.Sequential(
            nn.Linear(3 * embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(
        self,
        tabular_embed: torch.Tensor,
        image_embed: torch.Tensor, 
        text_embed: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        # Type-specific fusion
        unimodal_output, unimodal_aux = self.unimodal_moe(
            tabular_embed, image_embed, text_embed, training
        )
        
        crossmodal_output, crossmodal_aux = self.crossmodal_moe(
            tabular_embed, image_embed, text_embed, training
        )
        
        synergy_output, synergy_aux = self.synergy_moe(
            tabular_embed, image_embed, text_embed, training
        )
        
        # Final fusion
        hierarchical_input = torch.cat([unimodal_output, crossmodal_output, synergy_output], dim=1)
        final_output = self.final_fusion(hierarchical_input)
        
        # Combine auxiliary outputs
        auxiliary = {
            'unimodal': unimodal_aux,
            'crossmodal': crossmodal_aux,
            'synergy': synergy_aux
        }
        
        return final_output, auxiliary
