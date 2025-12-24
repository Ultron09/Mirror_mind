"""
Baseline Models for God Killer Test Suite

Implements Transformer, LSTM, RNN, and CNN baselines
that can be directly compared to MirrorMind v7.0.

All models use same interface for fair comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict


class TransformerBaseline(nn.Module):
    """Transformer-based model for sequence reasoning tasks."""
    
    def __init__(self, input_size: int = 30*30, hidden_dim: int = 256, num_layers: int = 4):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embed = nn.Linear(input_size, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=512,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.output = nn.Linear(hidden_dim, input_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size = x.shape[0]
        
        # Flatten and embed
        x = x.view(batch_size, -1)
        x = self.embed(x)  # (batch, hidden_dim)
        x = x.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Transformer
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        
        # Output
        x = self.output(x)
        x = x.view(batch_size, 1, 30, 30)
        
        return x


class LSTMBaseline(nn.Module):
    """LSTM-based model for sequence reasoning."""
    
    def __init__(self, input_size: int = 30*30, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        
        # Input processing
        self.embed = nn.Linear(input_size, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Output
        self.output = nn.Linear(hidden_dim, input_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size = x.shape[0]
        
        # Flatten and embed
        x = x.view(batch_size, -1)
        x = self.embed(x)
        x = x.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Take last hidden state
        
        # Output
        x = self.output(x)
        x = x.view(batch_size, 1, 30, 30)
        
        return x


class RNNBaseline(nn.Module):
    """RNN-based model for sequence reasoning."""
    
    def __init__(self, input_size: int = 30*30, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        
        # Input processing
        self.embed = nn.Linear(input_size, hidden_dim)
        
        # GRU layers (modern RNN)
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Output
        self.output = nn.Linear(hidden_dim, input_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size = x.shape[0]
        
        # Flatten and embed
        x = x.view(batch_size, -1)
        x = self.embed(x)
        x = x.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # RNN
        rnn_out, _ = self.rnn(x)
        x = rnn_out[:, -1, :]  # Take last hidden state
        
        # Output
        x = self.output(x)
        x = x.view(batch_size, 1, 30, 30)
        
        return x


class CNNBaseline(nn.Module):
    """CNN-based model for vision tasks."""
    
    def __init__(self, input_size: int = 30):
        super().__init__()
        self.input_size = input_size
        
        # Encoder (downsampling)
        self.enc1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Bottleneck
        self.bottleneck = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Decoder (upsampling)
        self.dec3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.dec1 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Encoder
        e1 = F.relu(self.enc1(x))
        e1_pool = F.max_pool2d(e1, 2)
        
        e2 = F.relu(self.enc2(e1_pool))
        e2_pool = F.max_pool2d(e2, 2)
        
        e3 = F.relu(self.enc3(e2_pool))
        e3_pool = F.max_pool2d(e3, 2)
        
        # Bottleneck
        b = F.relu(self.bottleneck(e3_pool))
        
        # Decoder
        d3 = F.interpolate(b, scale_factor=2, mode='nearest')
        d3 = F.relu(self.dec3(d3))
        
        d2 = F.interpolate(d3, scale_factor=2, mode='nearest')
        d2 = F.relu(self.dec2(d2))
        
        d1 = F.interpolate(d2, scale_factor=2, mode='nearest')
        d1 = self.dec1(d1)
        
        return d1


class EWCBaseline(nn.Module):
    """Baseline with Elastic Weight Consolidation (without consciousness)."""
    
    def __init__(self, input_size: int = 30*30, hidden_dim: int = 256):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        
        # Simple MLP backbone
        self.embed = nn.Linear(input_size, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, input_size)
        
        # EWC: store fisher information
        self.register_buffer('fisher', None)
        self.register_buffer('optimal_params', None)
        self.ewc_lambda = 0.4
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        
        x = F.relu(self.embed(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        
        x = x.view(batch_size, 1, 30, 30)
        return x
    
    def compute_ewc_loss(self) -> torch.Tensor:
        """Compute EWC penalty."""
        if self.fisher is None or self.optimal_params is None:
            return torch.tensor(0.0)
        
        ewc_loss = 0.0
        for (name, param), fisher_diag, optimal_param in zip(
            self.named_parameters(),
            self.fisher,
            self.optimal_params
        ):
            ewc_loss += (fisher_diag * (param - optimal_param).pow(2)).sum()
        
        return self.ewc_lambda * ewc_loss
    
    def consolidate_weights(self):
        """Consolidate weights after task learning."""
        # Simple consolidation: save current parameters
        self.optimal_params = [p.clone() for p in self.parameters()]
        
        # Estimate fisher information (simplified)
        fisher_diag = []
        for p in self.parameters():
            fisher_diag.append(torch.ones_like(p) * 0.1)
        self.fisher = fisher_diag


class SIBaseline(nn.Module):
    """Baseline with Synaptic Intelligence (without consciousness)."""
    
    def __init__(self, input_size: int = 30*30, hidden_dim: int = 256):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        
        # Simple MLP backbone
        self.embed = nn.Linear(input_size, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, input_size)
        
        # SI: store importance weights
        self.register_buffer('importance', None)
        self.register_buffer('prev_params', None)
        self.si_lambda = 0.4
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        
        x = F.relu(self.embed(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        
        x = x.view(batch_size, 1, 30, 30)
        return x
    
    def compute_si_loss(self) -> torch.Tensor:
        """Compute SI penalty."""
        if self.importance is None or self.prev_params is None:
            return torch.tensor(0.0)
        
        si_loss = 0.0
        for (name, param), importance, prev_param in zip(
            self.named_parameters(),
            self.importance,
            self.prev_params
        ):
            si_loss += (importance * (param - prev_param).pow(2)).sum()
        
        return self.si_lambda * si_loss
    
    def update_importance(self, learning_rate: float = 0.01):
        """Update importance weights."""
        if self.prev_params is None:
            self.prev_params = [p.clone() for p in self.parameters()]
            self.importance = [torch.zeros_like(p) for p in self.parameters()]
        else:
            # Update importance based on parameter changes
            for imp, param, prev_param in zip(
                self.importance,
                self.parameters(),
                self.prev_params
            ):
                param_change = param - prev_param
                imp += (param_change.abs() / (learning_rate + 1e-8))
            
            # Update previous params
            for i, param in enumerate(self.parameters()):
                self.prev_params[i] = param.clone()


class BaselineFactory:
    """Factory for creating baseline models."""
    
    @staticmethod
    def create_transformer(input_size: int = 30*30, **kwargs) -> nn.Module:
        """Create Transformer baseline."""
        return TransformerBaseline(input_size=input_size, **kwargs)
    
    @staticmethod
    def create_lstm(input_size: int = 30*30, **kwargs) -> nn.Module:
        """Create LSTM baseline."""
        return LSTMBaseline(input_size=input_size, **kwargs)
    
    @staticmethod
    def create_rnn(input_size: int = 30*30, **kwargs) -> nn.Module:
        """Create RNN baseline."""
        return RNNBaseline(input_size=input_size, **kwargs)
    
    @staticmethod
    def create_cnn(input_size: int = 30, **kwargs) -> nn.Module:
        """Create CNN baseline."""
        return CNNBaseline(input_size=input_size, **kwargs)
    
    @staticmethod
    def create_ewc(input_size: int = 30*30, **kwargs) -> nn.Module:
        """Create EWC baseline."""
        return EWCBaseline(input_size=input_size, **kwargs)
    
    @staticmethod
    def create_si(input_size: int = 30*30, **kwargs) -> nn.Module:
        """Create SI baseline."""
        return SIBaseline(input_size=input_size, **kwargs)
    
    @staticmethod
    def get_all_baseline_names() -> list:
        """Get all available baseline names."""
        return ['transformer', 'lstm', 'rnn', 'cnn', 'ewc', 'si']
    
    @staticmethod
    def create(name: str, **kwargs) -> nn.Module:
        """Create baseline by name."""
        creators = {
            'transformer': BaselineFactory.create_transformer,
            'lstm': BaselineFactory.create_lstm,
            'rnn': BaselineFactory.create_rnn,
            'cnn': BaselineFactory.create_cnn,
            'ewc': BaselineFactory.create_ewc,
            'si': BaselineFactory.create_si
        }
        
        if name not in creators:
            raise ValueError(f"Unknown baseline: {name}")
        
        return creators[name](**kwargs)


if __name__ == "__main__":
    # Test baselines
    print("Testing baseline models...\n")
    
    x = torch.randn(4, 1, 30, 30)
    
    for name in BaselineFactory.get_all_baseline_names():
        model = BaselineFactory.create(name)
        output = model(x)
        print(f"{name:15s}: input {x.shape} -> output {output.shape}")
