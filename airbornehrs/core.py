"""
Core Adaptive Meta-Learning Framework
======================================

Contains the base AdaptiveFramework and IntrospectionModule for continuous learning.
This is the production-ready core that users integrate into their applications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional, Any, Callable
import numpy as np
from collections import deque
import json
from pathlib import Path
from abc import ABC, abstractmethod
import logging
from datetime import datetime


# ==================== CONFIGURATION ====================

@dataclass
class AdaptiveFrameworkConfig:
    """
    Configuration for the adaptive meta-learning framework.
    """
    # Model architecture
    model_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1
    
    # Learning parameters
    learning_rate: float = 1e-3
    meta_learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 10
    
    # Adaptation parameters
    weight_adaptation_lr: float = 1e-5
    bias_adaptation_lr: float = 1e-5
    
    # Framework parameters
    feedback_buffer_size: int = 10000
    evaluation_frequency: int = 100
    adaptation_threshold: float = 0.05
    
    # Meta-learning (optimization cycle)
    inner_loop_steps: int = 5
    outer_loop_steps: int = 1
    
    # Logging
    log_frequency: int = 50
    checkpoint_frequency: int = 500


@dataclass
class PerformanceSnapshot:
    """Feedback snapshot from training/inference"""
    input_data: torch.Tensor
    output: torch.Tensor
    target: torch.Tensor
    reward: float
    loss: float
    timestamp: float
    episode: int
    
    def to_device(self, device):
        """Move all tensors to device"""
        self.input_data = self.input_data.to(device)
        self.output = self.output.to(device)
        self.target = self.target.to(device)
        return self


@dataclass
class MetricsSnapshot:
    """Snapshot of model performance metrics"""
    timestamp: float
    episode: int
    step: int
    train_loss: float
    eval_loss: float
    adaptation_needed: bool
    weight_adaptation_magnitude: float
    bias_adaptation_magnitude: float
    learning_efficiency: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== INTROSPECTION MODULE ====================

class IntrospectionModule(nn.Module):
    """
    State monitoring and introspection layer for performance calibration.
    """
    
    def __init__(self, config: AdaptiveFrameworkConfig):
        super().__init__()
        self.config = config
        
        # Main transformer layers
        self.embedding = nn.Linear(config.model_dim, config.model_dim)
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.model_dim,
                nhead=config.num_heads,
                dim_feedforward=config.ff_dim,
                dropout=config.dropout,
                batch_first=True
            ) for _ in range(config.num_layers)
        ])
        
        # Introspection probe: monitors internal state
        self.state_monitor = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim // 2),
            nn.ReLU(),
            nn.Linear(config.model_dim // 2, 1)
        )
        
        # Output heads
        self.output_head = nn.Linear(config.model_dim, config.model_dim)
        
        # Uncertainty estimation (Outputs Log Variance for Gaussian NLL)
        self.uncertainty_head = nn.Linear(config.model_dim, 1)
        
    def forward(self, x: torch.Tensor, return_internals: bool = False):
        """
        Forward pass with optional introspection.
        
        Returns:
            output: Model output
            log_var: Uncertainty estimate (Log Variance)
            internals: (optional) Internal state dictionary
        """
        batch_size, seq_len, _ = x.shape
        
        # Embed input
        x = self.embedding(x)
        internals = {'embeddings': x.clone()}
        
        # Apply transformer layers
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x)
            internals[f'layer_{i}'] = x.clone()
        
        # Recursive state monitoring: analyze patterns in learned representations
        introspection_signal = self.state_monitor(x)
        internals['introspection'] = introspection_signal
        
        # Generate output
        output = self.output_head(x)
        
        # Uncertainty estimation (Log Variance)
        # We clamp it or treat it directly as log variance
        log_var = self.uncertainty_head(x)
        
        if return_internals:
            return output, log_var, internals
        return output, log_var


# ==================== PERFORMANCE MONITOR ====================

class PerformanceMonitor:
    """
    Meta-controller component for dynamic adaptation.
    """
    
    def __init__(self, model: IntrospectionModule, config: AdaptiveFrameworkConfig, device):
        self.model = model
        self.config = config
        self.device = device
        self.logger = logging.getLogger('PerformanceMonitor')
        
        self.weight_adaptation_history = deque(maxlen=1000)
        self.bias_adaptation_history = deque(maxlen=1000)
        
    def compute_layer_importance(self, activations: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute importance scores for each layer."""
        importance_scores = {}
        
        for layer_name, activation in activations.items():
            if 'layer_' in layer_name:
                mean_activation = activation.abs().mean()
                std_activation = activation.std()
                activation_sparsity = (activation.abs() < 0.01).float().mean()
                
                importance = mean_activation * std_activation * (1 - activation_sparsity)
                importance_scores[layer_name] = float(importance)
        
        return importance_scores
    
    def compute_gradient_statistics(self, loss: torch.Tensor) -> Dict[str, float]:
        """Compute gradient statistics for learning diagnostics"""
        loss.backward(retain_graph=True)
        
        grad_stats = {
            'mean_grad': 0,
            'max_grad': 0,
            'min_grad': 0,
            'grad_variance': 0
        }
        
        total_grad_norm = 0
        num_params = 0
        
        for param in self.model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item()
                num_params += 1
        
        grad_stats['mean_grad'] = total_grad_norm / max(num_params, 1)
        
        return grad_stats
    
    def adapt_weights(self, 
                      current_loss: float, 
                      previous_loss: float,
                      activations: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        """
        Adapt model weights based on loss trajectory and layer importance.
        """
        loss_improvement = (previous_loss - current_loss) / (previous_loss + 1e-9)
        
        adaptation_magnitude = 0.0
        
        if loss_improvement < self.config.adaptation_threshold:
            importance_scores = self.compute_layer_importance(activations)
            
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        layer_importance = 0.5
                        for layer_name, importance in importance_scores.items():
                            if layer_name in name:
                                layer_importance = importance
                        
                        adaptation = torch.randn_like(param) * self.config.weight_adaptation_lr * layer_importance
                        param.data += adaptation
                        adaptation_magnitude += adaptation.abs().mean().item()
        
        self.weight_adaptation_history.append(adaptation_magnitude)
        
        return adaptation_magnitude, 0.0


# ==================== FEEDBACK BUFFER ====================

class FeedbackBuffer:
    """Experience replay buffer for online learning."""
    
    def __init__(self, config: AdaptiveFrameworkConfig, device):
        self.config = config
        self.device = device
        self.buffer = deque(maxlen=config.feedback_buffer_size)
        self.episode_count = 0
        self.step_count = 0
        
    def add(self, input_data, output, target, reward, loss):
        """Add feedback to buffer"""
        snapshot = PerformanceSnapshot(
            input_data=input_data.cpu().clone(),
            output=output.cpu().clone(),
            target=target.cpu().clone(),
            reward=reward,
            loss=loss,
            timestamp=datetime.now().timestamp(),
            episode=self.episode_count
        )
        self.buffer.append(snapshot)
        self.step_count += 1
        
    def sample_recent(self, batch_size: int) -> Optional[List[PerformanceSnapshot]]:
        """Sample recent batch (on-policy)"""
        if len(self.buffer) < batch_size:
            return None
        return list(self.buffer)[-batch_size:]


# ==================== ADAPTIVE FRAMEWORK ====================

class AdaptiveFramework:
    """
    Production-ready adaptive meta-learning framework.
    """
    
    def __init__(self, config: AdaptiveFrameworkConfig, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.device = device
        self.config = config
        self.logger = self._setup_logging()
        
        # Core components
        self.model = IntrospectionModule(config).to(device)
        self.monitor = PerformanceMonitor(self.model, config, device)
        self.feedback_buffer = FeedbackBuffer(config, device)
        
        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)
        
        # Metrics
        self.metrics_history: List[MetricsSnapshot] = []
        self.loss_history = deque(maxlen=config.evaluation_frequency)
        
        self.step_count = 0
        self.epoch_count = 0
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('AdaptiveFramework')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            logger.addHandler(handler)
        return logger
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for inference."""
        x = x.to(self.device)
        with torch.no_grad():
            output, log_var = self.model(x, return_internals=False)
        return output, log_var
    
    def train_step(self, 
                   input_data: torch.Tensor,
                   target: torch.Tensor) -> Dict[str, float]:
        """
        Single training step with introspection and adaptation.
        """
        input_data = input_data.to(self.device)
        target = target.to(self.device)
        
        # 1. Forward pass
        # We get 'log_var' (log variance), not 'uncertainty'
        output, log_var, internals = self.model(input_data, return_internals=True)
        
        # 2. Compute Loss: Gaussian Negative Log Likelihood
        precision = torch.exp(-log_var)
        mse = (output - target) ** 2
        loss = torch.mean(0.5 * (log_var + mse * precision))
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        metrics = {
            'loss': loss.item(),
            # FIX: Use log_var, which is the variable we actually have
            'uncertainty_mean': log_var.mean().item()
        }
        
        self.loss_history.append(loss.item())
        
        if self.step_count % self.config.evaluation_frequency == 0:
            if len(self.loss_history) > 1:
                current_loss = np.mean(list(self.loss_history)[-10:])
                previous_loss = np.mean(list(self.loss_history)[-20:-10])
                
                weight_adapt_mag, bias_adapt_mag = self.monitor.adapt_weights(
                    current_loss, previous_loss, internals
                )
                
                metrics['weight_adaptation_magnitude'] = weight_adapt_mag
                metrics['bias_adaptation_magnitude'] = bias_adapt_mag
        
        self.step_count += 1
        
        return metrics
    
    def evaluate(self, 
                 input_data: torch.Tensor,
                 target: torch.Tensor) -> Dict[str, float]:
        """Evaluate on validation data."""
        input_data = input_data.to(self.device)
        target = target.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            output, log_var = self.model(input_data, return_internals=False)
            loss = F.mse_loss(output, target)
        self.model.train()
        
        return {
            'eval_loss': loss.item(),
            'uncertainty_mean': log_var.mean().item()
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'config': self.config,
            'step_count': self.step_count
        }, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.step_count = checkpoint.get('step_count', 0)
        self.logger.info(f"Checkpoint loaded from {path}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get summary of metrics"""
        if not self.loss_history:
            return {}
        return {'avg_recent_loss': np.mean(self.loss_history)}