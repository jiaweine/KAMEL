"""
KAN (Kolmogorov-Arnold Network) layers implementation.
Adapted from EasyTSF KAN layers for tabular learning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


def B_batch(x, grid, k=0, extend=True, device='cuda'):

    def extend_grid(grid, k_extend=0):
        h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)
        for i in range(k_extend):
            grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
            grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
        grid = grid.to(device)
        return grid

    if extend:
        grid = extend_grid(grid, k_extend=k)

    grid = grid.unsqueeze(dim=2).to(device)
    x = x.unsqueeze(dim=1).to(device)

    if k == 0:
        value = (x >= grid[:, :-1]) * (x < grid[:, 1:])
    else:
        B_km1 = B_batch(x[:, 0], grid=grid[:, :, 0], k=k - 1, extend=False, device=device)
        value = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * B_km1[:, :-1] + \
                (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * B_km1[:, 1:]
    return value


def coef2curve(x_eval, grid, coef, k, device="cuda"):

    if coef.dtype != x_eval.dtype:
        coef = coef.to(x_eval.dtype)
    y_eval = torch.einsum('ij,ijk->ik', coef, B_batch(x_eval, grid, k, device=device))
    return y_eval


def curve2coef(x_eval, y_eval, grid, k, device="cuda"):

    mat = B_batch(x_eval, grid, k, device=device).permute(0, 2, 1)
    coef = torch.linalg.lstsq(mat.to(device), y_eval.unsqueeze(dim=2).to(device),
                              driver='gelsy' if device == 'cpu' else 'gels').solution[:, :, 0]
    return coef.to(device)


class KANLayer(nn.Module):

    
    def __init__(
        self, 
        in_dim: int = 3, 
        out_dim: int = 2, 
        num: int = 5, 
        k: int = 3, 
        noise_scale: float = 0.1,
        scale_base: float = 1.0, 
        scale_sp: float = 1.0,
        base_fun = nn.SiLU(), 
        grid_eps: float = 0.02, 
        grid_range: list = [-1, 1], 
        sp_trainable: bool = True, 
        sb_trainable: bool = True,
        device: str = 'cuda'
    ):

        super(KANLayer, self).__init__()
        
        self.size = size = out_dim * in_dim
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k

        # Grid initialization
        self.grid = torch.einsum('i,j->ij', torch.ones(size, device=device),
                                 torch.linspace(grid_range[0], grid_range[1], steps=num + 1, device=device))
        self.grid = nn.Parameter(self.grid).requires_grad_(False)
        
        # Noise for initialization
        noises = (torch.rand(size, self.grid.shape[1], device=device) - 1/2) * noise_scale / num
        
        # Coefficients
        self.coef = nn.Parameter(curve2coef(self.grid, noises, self.grid, k, device))
        
        # Scales
        if isinstance(scale_base, float):
            self.scale_base = nn.Parameter(torch.ones(size, device=device) * scale_base).requires_grad_(sb_trainable)
        else:
            self.scale_base = nn.Parameter(torch.FloatTensor(scale_base).to(device)).requires_grad_(sb_trainable)
        
        self.scale_sp = nn.Parameter(torch.ones(size, device=device) * scale_sp).requires_grad_(sp_trainable)
        
        self.base_fun = base_fun
        self.mask = nn.Parameter(torch.ones(size, device=device)).requires_grad_(False)
        self.grid_eps = grid_eps
        self.weight_sharing = torch.arange(size, device=device)
        self.device = device

    def forward(self, x):

        batch = x.shape[0]
        
        # Expand input
        x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, device=self.device)).reshape(batch, self.size).permute(1, 0)
        
        # Base activation
        base = self.base_fun(x).permute(1, 0)
        
        # Spline activation
        y = coef2curve(x_eval=x, grid=self.grid[self.weight_sharing], coef=self.coef[self.weight_sharing], 
                      k=self.k, device=self.device)
        y = y.permute(1, 0)
        
        # Combine base and spline
        y = self.scale_base.unsqueeze(dim=0) * base + self.scale_sp.unsqueeze(dim=0) * y
        y = self.mask[None, :] * y
        
        # Sum across input dimensions
        y = torch.sum(y.reshape(batch, self.out_dim, self.in_dim), dim=2)
        
        return y


class FastKANLayer(nn.Module):

    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 8,
        base_activation = nn.SiLU(),
        grid_range: Tuple[float, float] = (-2., 2.),
        device: str = 'cuda'
    ):
        super(FastKANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.base_activation = base_activation
        self.device = device
        
        # Initialize grid
        grid = torch.linspace(grid_range[0], grid_range[1], grid_size, device=device)
        self.register_buffer('grid', grid)
        
        # RBF parameters
        self.rbf_weight = nn.Parameter(torch.randn(in_features, out_features, grid_size, device=device) * 0.1)
        self.spline_weight = nn.Parameter(torch.randn(in_features, out_features, device=device) * 0.1)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features, device=device))
        
        # RBF width
        self.register_buffer('rbf_width', torch.tensor(2.0 / (grid_size - 1), device=device))
        
    def forward(self, x):

        batch_size = x.shape[0]
        
        # Base activation
        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        # RBF activation
        x_expanded = x.unsqueeze(-1)  # (batch, in_features, 1)
        grid_expanded = self.grid.unsqueeze(0).unsqueeze(0)  # (1, 1, grid_size)
        
        # Compute RBF values
        rbf_output = torch.exp(-((x_expanded - grid_expanded) / self.rbf_width) ** 2)  # (batch, in_features, grid_size)
        
        # Weight RBF outputs
        # rbf_output: (batch, in_features, grid_size)
        # rbf_weight: (in_features, out_features, grid_size)
        weighted_rbf = torch.einsum('big,iog->bio', rbf_output, self.rbf_weight)  # (batch, in_features, out_features)
        # Sum over input features and apply spline weights
        summed_rbf = weighted_rbf.sum(dim=1)  # (batch, out_features)
        spline_output = summed_rbf * self.spline_weight.sum(dim=0)  # (batch, out_features)
        
        return base_output + spline_output


class ChebyKANLayer(nn.Module):

    
    def __init__(self, input_dim: int, output_dim: int, degree: int):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):

        # Normalize input to [-1, 1] using tanh
        x = torch.tanh(x)
        
        # Expand and repeat input
        x = x.view((-1, self.inputdim, 1)).expand(-1, -1, self.degree + 1)
        
        # Apply Chebyshev transform
        x = x.acos()
        x *= self.arange
        x = x.cos()
        
        # Compute Chebyshev interpolation
        y = torch.einsum("bid,iod->bo", x, self.cheby_coeffs)
        y = y.view(-1, self.outdim)
        
        return y


class WaveKANLayer(nn.Module):

    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        wavelet_type: str = 'mexican_hat',
        device: str = "cuda"
    ):
        super(WaveKANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type

        # Wavelet parameters
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))
        
        # Weights
        self.weight1 = nn.Parameter(torch.Tensor(out_features, in_features))
        self.wavelet_weights = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))

        # Base activation
        self.base_activation = nn.SiLU()

    def wavelet_transform(self, x):
        """Apply wavelet transformation."""
        if x.dim() == 2:
            x_expanded = x.unsqueeze(1)
        else:
            x_expanded = x

        translation_expanded = self.translation.unsqueeze(0).expand(x.size(0), -1, -1)
        scale_expanded = self.scale.unsqueeze(0).expand(x.size(0), -1, -1)
        x_scaled = (x_expanded - translation_expanded) / scale_expanded

        if self.wavelet_type == 'mexican_hat':
            term1 = ((x_scaled ** 2) - 1)
            term2 = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = (2 / (math.sqrt(3) * math.pi ** 0.25)) * term1 * term2
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
            
        elif self.wavelet_type == 'morlet':
            omega0 = 5.0  # Central frequency
            real = torch.cos(omega0 * x_scaled)
            envelope = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = envelope * real
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
            
        elif self.wavelet_type == 'dog':
            # Derivative of Gaussian Wavelet
            dog = -x_scaled * torch.exp(-0.5 * x_scaled ** 2)
            wavelet = dog
            wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
            wavelet_output = wavelet_weighted.sum(dim=2)
        else:
            raise ValueError(f"Unsupported wavelet type: {self.wavelet_type}")

        return wavelet_output

    def forward(self, x):
        wavelet_output = self.wavelet_transform(x)
        base_output = F.linear(x, self.weight1)
        combined_output = wavelet_output + base_output
        
        return combined_output


# KAN Layer factory for easy instantiation
class KANLayerFactory:
    
    @staticmethod
    def create_layer(
        layer_type: str,
        in_features: int,
        out_features: int,
        **kwargs
    ) -> nn.Module:

        if layer_type == 'kan':
            return KANLayer(
                in_dim=in_features, 
                out_dim=out_features,
                num=kwargs.get('grid_size', 8),
                k=kwargs.get('spline_order', 3),
                device=kwargs.get('device', 'cuda')
            )
        elif layer_type == 'fast_kan':
            return FastKANLayer(
                in_features=in_features,
                out_features=out_features,
                grid_size=kwargs.get('grid_size', 8),
                device=kwargs.get('device', 'cuda')
            )
        elif layer_type == 'cheby_kan':
            return ChebyKANLayer(
                input_dim=in_features,
                output_dim=out_features,
                degree=kwargs.get('degree', 3)
            )
        elif layer_type == 'wave_kan':
            return WaveKANLayer(
                in_features=in_features,
                out_features=out_features,
                wavelet_type=kwargs.get('wavelet_type', 'mexican_hat'),
                device=kwargs.get('device', 'cuda')
            )
        else:
            raise ValueError(f"Unknown KAN layer type: {layer_type}")
