# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:22:14 2025

@author: Admin
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union
from einops import rearrange, repeat
from dataclasses import dataclass

from scans import selective_scan


# %%
@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    scan_mode: str = 'cumsum'
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


# %%
class MambaSSM(nn.Module):
    
    def __init__(self, args: ModelArgs, input_dimension: int, output_length: int):
        
        """MAMBA SSM model for time series forecasting."""
        super().__init__()
        self.args = args
        self.output_length = output_length

        # Input layer: Expecting input shape (batch_size, historic_horizon, 7)
        self.input_layer = nn.Linear(input_dimension, args.d_model)
        
        # Mamba layers
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)

        # Output layer: Predicting 1 feature for each of the forecast_horizon time steps
        self.output_layer = nn.Linear(args.d_model, 1)

    def forward(self, x):
        """
        Args:
            x: A tensor of shape (batch_size, historic_horizon, 7)
        Returns:
            forecast: A tensor of shape (batch_size, forecast_horizon, 1)
        """
        # Step 1: Transform input to the appropriate shape
        x = self.input_layer(x)  # shape: (batch_size, historic_horizon, d_model)
        
        # Step 2: Process the sequence through the MAMBA blocks
        for layer in self.layers:
            x = layer(x)

        # Step 3: Normalize the output
        x = self.norm_f(x)

        # Step 4: Output the forecast (forecast_horizon x 1 feature)
        forecast = self.output_layer(x[:, -self.output_length:, :])  # Use only the forecast horizon
        
        return forecast


class ResidualBlock(nn.Module):
    
    def __init__(self, args: ModelArgs):
        
        """A residual block with MambaBlock and RMSNorm."""
        super().__init__()
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)
        
    def forward(self, x):
        return self.mixer(self.norm(x)) + x
            

class MambaBlock(nn.Module):
    
    def __init__(self, args: ModelArgs):
        
        """A single Mamba Block with convolutions and state-space processing."""
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)
        
    def forward(self, x):
        
        """Forward pass for the Mamba block."""
        (b, l, d) = x.shape
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x)
        y = self.ssm(x)
        
        y = y * F.silu(res)
        
        return self.out_proj(y)

    def ssm(self, x):
        
        """Selective State-Space Model (SSM) forward pass."""
        (d_in, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        
        return selective_scan(x, delta, A, B, C, D, mode=self.args.scan_mode)


class RMSNorm(nn.Module):
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# %%
def create_model(data, forecast_horizon, device):
    
    input_dimension = len(data.columns)
    output_length = forecast_horizon
    
    arguments  = ModelArgs(d_model = 6, n_layer = 4)
    
    model = MambaSSM(arguments, input_dimension, output_length).to(device)
    print("Mamba SSM model successfully initialized.")
    return model

