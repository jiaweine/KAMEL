"""
KAN-based encoders for multimodal tabular learning.
Implements three specialized encoders for tabular, image, and text modalities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from .kan_layers import KANLayerFactory


class TabularKANEncoder(nn.Module):
    """
    KAN-based encoder for tabular (numerical) features.
    Directly processes numerical features with KAN layers.
    """
    
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        kan_type: str = 'fast_kan',
        dropout: float = 0.1,
        layer_norm: bool = True,
        device: str = 'cuda'
    ):

        super(TabularKANEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.kan_type = kan_type
        self.device = device
        
        # Default architecture
        if hidden_dims is None:
            hidden_dims = [128]
        
        # Build KAN layers
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            # KAN layer
            kan_layer = KANLayerFactory.create_layer(
                layer_type=kan_type,
                in_features=in_dim,
                out_features=hidden_dim,
                device=device
            )
            layers.append(kan_layer)
            
            # Layer normalization
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            in_dim = hidden_dim
        
        # Final projection to embed_dim
        final_kan = KANLayerFactory.create_layer(
            layer_type=kan_type,
            in_features=in_dim,
            out_features=embed_dim,
            device=device
        )
        layers.append(final_kan)
        
        if layer_norm:
            layers.append(nn.LayerNorm(embed_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # Ensure all layers are on the correct device
        self.to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Ensure input is on the correct device
        x = x.to(self.device)
        return self.encoder(x)


class ImageKANEncoder(nn.Module):

    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 32, 32),  # (C, H, W)
        embed_dim: int = 64,
        cnn_channels: List[int] = [16, 32],
        kan_hidden_dims: List[int] = [128],
        kan_type: str = 'fast_kan',
        dropout: float = 0.1,
        device: str = 'cuda'
    ):

        super(ImageKANEncoder, self).__init__()
        
        self.input_shape = input_shape
        self.embed_dim = embed_dim
        self.kan_type = kan_type
        self.device = device
        
        C, H, W = input_shape
        
        # CNN feature extractor
        cnn_layers = []
        in_channels = C
        
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),  # Use BatchNorm instead of LayerNorm for CNN
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ])
            in_channels = out_channels
            H, W = H // 2, W // 2
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Calculate flattened dimension after CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            cnn_output = self.cnn(dummy_input)
            cnn_flat_dim = cnn_output.view(1, -1).shape[1]
        
        # KAN layers for final encoding
        kan_layers = []
        in_dim = cnn_flat_dim
        
        for hidden_dim in kan_hidden_dims:
            kan_layer = KANLayerFactory.create_layer(
                layer_type=kan_type,
                in_features=in_dim,
                out_features=hidden_dim,
                device=device
            )
            kan_layers.extend([
                kan_layer,
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # Final projection
        final_kan = KANLayerFactory.create_layer(
            layer_type=kan_type,
            in_features=in_dim,
            out_features=embed_dim,
            device=device
        )
        kan_layers.extend([
            final_kan,
            nn.LayerNorm(embed_dim)
        ])
        
        self.kan_encoder = nn.Sequential(*kan_layers)
        
        # Ensure all layers are on the correct device
        self.to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Ensure input is on the correct device
        x = x.to(self.device)
        
        # CNN feature extraction
        cnn_features = self.cnn(x)
        
        # Flatten
        flat_features = cnn_features.view(cnn_features.size(0), -1)
        
        # KAN encoding
        encoded = self.kan_encoder(flat_features)
        
        return encoded


class TextKANEncoder(nn.Module):

    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        seq_embed_dim: int = 128,
        max_seq_length: int = 512,
        kan_hidden_dims: List[int] = [128],
        kan_type: str = 'fast_kan',
        conv_kernel_size: int = 3,
        dropout: float = 0.1,
        device: str = 'cuda'
    ):

        super(TextKANEncoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_embed_dim = seq_embed_dim
        self.max_seq_length = max_seq_length
        self.kan_type = kan_type
        self.device = device
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, seq_embed_dim, padding_idx=0)
        
        # Positional encoding (optional)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_length, seq_embed_dim) * 0.1
        )
        
        # 1D Convolution for sequence modeling  
        self.conv1d = nn.Conv1d(
            in_channels=seq_embed_dim,
            out_channels=seq_embed_dim,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # KAN layers for final encoding
        kan_layers = []
        in_dim = seq_embed_dim
        
        for hidden_dim in kan_hidden_dims:
            kan_layer = KANLayerFactory.create_layer(
                layer_type=kan_type,
                in_features=in_dim,
                out_features=hidden_dim,
                device=device
            )
            kan_layers.extend([
                kan_layer,
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # Final projection
        final_kan = KANLayerFactory.create_layer(
            layer_type=kan_type,
            in_features=in_dim,
            out_features=embed_dim,
            device=device
        )
        kan_layers.extend([
            final_kan,
            nn.LayerNorm(embed_dim)
        ])
        
        self.kan_encoder = nn.Sequential(*kan_layers)
        
        # Ensure all layers are on the correct device
        self.to(device)
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # Ensure inputs are on the correct device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            
        batch_size, seq_length = input_ids.shape
        
        # Token embeddings
        embeddings = self.token_embedding(input_ids)
        
        # Add positional encoding
        if seq_length <= self.max_seq_length:
            embeddings += self.pos_encoding[:, :seq_length, :]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            embeddings = embeddings * mask
        
        # 1D convolution (transpose for conv1d: batch, channels, sequence)
        conv_input = embeddings.transpose(1, 2)  # (batch, embed_dim, seq_length)
        conv_output = F.relu(self.conv1d(conv_input))
        
        # Global pooling
        pooled = self.global_pool(conv_output).squeeze(-1)  # (batch, embed_dim)
        
        # KAN encoding
        encoded = self.kan_encoder(pooled)
        
        return encoded


class UnifiedKANEncoders(nn.Module):

    
    def __init__(
        self,
        tabular_config: Dict,
        image_config: Dict,
        text_config: Dict,
        embed_dim: int = 64,
        device: str = 'cuda'
    ):

        super(UnifiedKANEncoders, self).__init__()
        
        self.embed_dim = embed_dim
        self.device = device
        
        # Tabular encoder
        self.tabular_encoder = TabularKANEncoder(
            embed_dim=embed_dim,
            device=device,
            **tabular_config
        )
        
        # Image encoder
        self.image_encoder = ImageKANEncoder(
            embed_dim=embed_dim,
            device=device,
            **image_config
        )
        
        # Text encoder
        self.text_encoder = TextKANEncoder(
            embed_dim=embed_dim,
            device=device,
            **text_config
        )
        
        # Ensure all encoders are on the correct device
        self.to(device)
        
    def forward(
        self,
        tabular_data: torch.Tensor,
        image_data: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Ensure all inputs are on the correct device
        tabular_data = tabular_data.to(self.device)
        image_data = image_data.to(self.device)
        text_input_ids = text_input_ids.to(self.device)
        if text_attention_mask is not None:
            text_attention_mask = text_attention_mask.to(self.device)
            
        # Encode each modality
        tabular_embed = self.tabular_encoder(tabular_data)
        image_embed = self.image_encoder(image_data)
        text_embed = self.text_encoder(text_input_ids, text_attention_mask)
        
        return tabular_embed, image_embed, text_embed
