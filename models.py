from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", bs=64, n_steps=17, output_dim=256):
        super().__init__()
        self.device = device
        self.bs = bs
        self.n_steps = n_steps
        self.repr_dim = 256

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        return torch.randn((self.bs, self.n_steps, self.repr_dim)).to(self.device)


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output

#Added param heavy implementation Encoder and Predictor

class Encoder(nn.Module):
    def __init__(self, repr_dim=256):
        super().__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, repr_dim)
        )
        
    def forward(self, x):
        if len(x.shape) == 5:
            B, T = x.shape[:2]
            x = x.reshape(B * T, *x.shape[2:])
            out = self.convnet(x)
            return out.reshape(B, T, -1)
        return self.convnet(x)

class Predictor(nn.Module):
    def __init__(self, repr_dim=256, action_dim=2):
        super().__init__()
        # Takes current state embedding and action, predicts next state embedding
        self.net = nn.Sequential(
            nn.Linear(repr_dim + action_dim, 512),
            nn.LayerNorm(512),  # LayerNorm for better training stability
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(True),
            nn.Linear(512, repr_dim)
        )
    
    def forward(self, state_embed, action):
        combined = torch.cat([state_embed, action], dim=-1)
        return self.net(combined)

class JEPAModel(nn.Module):
    def __init__(self, repr_dim=256):
        super().__init__()
        self.repr_dim = repr_dim
        self.encoder = Encoder(repr_dim=repr_dim)
        self.predictor = Predictor(repr_dim=repr_dim)
        
    def forward(self, states, actions):
        """
        Args:
            states: [B, T, Ch, H, W]  - Batch, Time, Channels, Height, Width
            actions: [B, T-1, 2]      - Batch, Time-1, Action_dim
            
        Returns:
            predictions: [B, T, D]     - Batch, Time, Repr_dim
        """
        B, T = states.shape[:2]
        predictions = []
        
        # Get initial state embedding
        curr_embed = self.encoder(states[:, 0])
        predictions.append(curr_embed)
        
        # Predict future embeddings
        for t in range(T-1):
            curr_embed = self.predictor(curr_embed, actions[:, t])
            predictions.append(curr_embed)
        
        return torch.stack(predictions, dim=1)  # [B, T, D]
    
    def get_target_embeddings(self, states):
        """Get embeddings for all states - used during training"""
        return self.encoder(states)


#Simpler Encoder less params

class SimpleEncoder(nn.Module):
    def __init__(self, repr_dim=256):
        super().__init__()
        self.convnet = nn.Sequential(
            # Single conv layer: 2 channels -> 32 channels
            nn.Conv2d(2, 32, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # Flatten and project to repr_dim
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, repr_dim)
        )
        
    def forward(self, x):
        if len(x.shape) == 5:
            B, T = x.shape[:2]
            x = x.reshape(B * T, *x.shape[2:])
            out = self.convnet(x)
            return out.reshape(B, T, -1)
        return self.convnet(x)

class SimplePredictor(nn.Module):
    def __init__(self, repr_dim=256, action_dim=2):
        super().__init__()
        # Single layer predictor
        self.net = nn.Sequential(
            nn.Linear(repr_dim + action_dim, repr_dim),
            nn.BatchNorm1d(repr_dim),
            nn.ReLU(True)
        )
    
    def forward(self, state_embed, action):
        combined = torch.cat([state_embed, action], dim=-1)
        return self.net(combined)

class SimpleJEPAModel(nn.Module):
    def __init__(self, repr_dim=364):
        super().__init__()
        self.repr_dim = repr_dim
        self.encoder = SimpleEncoder(repr_dim=repr_dim)
        self.predictor = SimplePredictor(repr_dim=repr_dim)
        
    def forward(self, states, actions):
        """
        Args:
            states: [B, T, Ch, H, W]  - Batch, Time, Channels, Height, Width
            actions: [B, T-1, 2]      - Batch, Time-1, Action_dim
            
        Returns:
            predictions: [B, T, D]     - Batch, Time, Repr_dim
        """
        B, T = states.shape[:2]
        predictions = []
        
        # Get initial state embedding
        curr_embed = self.encoder(states[:, 0])
        predictions.append(curr_embed)
        
        # Predict future embeddings
        for t in range(T-1):
            curr_embed = self.predictor(curr_embed, actions[:, t])
            predictions.append(curr_embed)
        
        return torch.stack(predictions, dim=1)

    def get_target_embeddings(self, states):
        """Get embeddings for all states - used during training"""
        return self.encoder(states)