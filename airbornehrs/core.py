"""
Core Adaptive Meta-Learning Framework (Optimized V2.1)
======================================================

Contains the base AdaptiveFramework and IntrospectionModule for continuous learning.
Includes High-Performance optimizations:
- torch.compile (Graph Compilation) -> With Windows Fallback
- norm_first=True (Pre-LN Stability)
- Reservoir Sampling (Long-term Memory)
- Robust Loss & NaN Guards
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional, Any
import numpy as np
import random
from collections import deque
from pathlib import Path
import logging
from datetime import datetime
import sys
import platform
import shutil
from airbornehrs.ewc import EWCHandler

# OPTIMIZATION: Use Tensor Cores on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

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
    
    # Optimization & Speed
    compile_model: bool = True       # Default to True, but will auto-disable if needed
    use_amp: bool = False            # Mixed Precision
    
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
    step: int
    loss: float
    uncertainty: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== INTROSPECTION MODULE ====================

class IntrospectionModule(nn.Module):
    """
    State monitoring and introspection layer for performance calibration.
    OPTIMIZED: Uses Pre-Norm (norm_first=True) for deep transformer stability.
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
                batch_first=True,
                norm_first=True,   # CRITICAL: Pre-Norm stabilizes deep training
                activation="gelu"  # Smoother than ReLU
            ) for _ in range(config.num_layers)
        ])
        
        # Introspection probe: monitors internal state
        self.state_monitor = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim // 2),
            nn.GELU(),
            nn.Linear(config.model_dim // 2, 1)
        )
        
        # Output heads
        self.output_head = nn.Linear(config.model_dim, config.model_dim)
        
        # Uncertainty estimation (Outputs Log Variance)
        self.uncertainty_head = nn.Linear(config.model_dim, 1)
        
    def forward(self, x: torch.Tensor, return_internals: bool = False):
        # Embed input
        x = self.embedding(x)
        internals = {}
        if return_internals:
            internals['embeddings'] = x.clone()
        
        # Apply transformer layers
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x)
            if return_internals:
                internals[f'layer_{i}'] = x.clone()
        
        # Recursive state monitoring
        if return_internals:
            introspection_signal = self.state_monitor(x)
            internals['introspection'] = introspection_signal
        
        # Generate output
        output = self.output_head(x)
        
        # Uncertainty estimation (Log Variance)
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
        
    def compute_layer_importance(self, activations: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute importance scores based on activation statistics."""
        importance_scores = {}
        for layer_name, activation in activations.items():
            if 'layer_' in layer_name:
                # Metric: High magnitude + High Variance = Important
                mean_activation = activation.abs().mean()
                std_activation = activation.std()
                # Avoid div by zero or nan
                if torch.isnan(std_activation): std_activation = torch.tensor(0.0)
                
                importance = mean_activation * std_activation
                importance_scores[layer_name] = float(importance.item())
        return importance_scores
    
    def adapt_weights(self, 
                      current_loss: float, 
                      previous_loss: float,
                      activations: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        """
        Adapt model weights in-place (Self-Modification).
        """
        loss_improvement = (previous_loss - current_loss)
        adaptation_magnitude = 0.0
        
        # If learning is stalling (low improvement), trigger self-modification
        if abs(loss_improvement) < self.config.adaptation_threshold:
            importance_scores = self.compute_layer_importance(activations)
            
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.requires_grad and 'transformer' in name:
                        layer_importance = 0.1 # Default base importance
                        for layer_name, importance in importance_scores.items():
                            if layer_name in name:
                                layer_importance = importance
                        
                        # Add controlled noise based on layer importance to break local minima
                        noise_scale = self.config.weight_adaptation_lr * layer_importance
                        adaptation = torch.randn_like(param) * noise_scale
                        
                        # Apply adaptation
                        param.data.add_(adaptation)
                        adaptation_magnitude += adaptation.abs().mean().item()
        
        self.weight_adaptation_history.append(adaptation_magnitude)
        return adaptation_magnitude, 0.0


# ==================== RESERVOIR BUFFER ====================

class FeedbackBuffer:
    """
    Robust Experience Replay Buffer using Reservoir Sampling (Algorithm R).
    Ensures long-term memory representation without unbounded growth.
    """
    
    def __init__(self, config: AdaptiveFrameworkConfig, device):
        self.capacity = config.feedback_buffer_size
        self.device = device
        
        # Fixed size list for reservoir
        self.buffer: List[PerformanceSnapshot] = []
        self.total_seen = 0 # Counter for Algorithm R
        
    def add(self, input_data, output, target, reward, loss):
        """Add feedback using Reservoir Sampling logic"""
        snapshot = PerformanceSnapshot(
            input_data=input_data.detach().cpu(),
            output=output.detach().cpu(),
            target=target.detach().cpu(),
            reward=reward,
            loss=loss,
            timestamp=datetime.now().timestamp(),
            episode=self.total_seen
        )
        
        # 1. Fill buffer until full
        if len(self.buffer) < self.capacity:
            self.buffer.append(snapshot)
        
        # 2. Once full, replace elements with probability k/n
        else:
            # Random index between 0 and total_seen
            replace_idx = random.randint(0, self.total_seen)
            if replace_idx < self.capacity:
                self.buffer[replace_idx] = snapshot
                
        self.total_seen += 1
        
    def sample_recent(self, batch_size: int) -> Optional[List[PerformanceSnapshot]]:
        """Sample batch from reservoir"""
        if len(self.buffer) < batch_size:
            return None
        return random.sample(self.buffer, batch_size)


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
        
        # 1. Core components
        self.model = IntrospectionModule(config).to(device)
        self.monitor = PerformanceMonitor(self.model, config, device)
        self.feedback_buffer = FeedbackBuffer(config, device)
        self.ewc = EWCHandler(self.model , ewc_lambda=0.4)
        # 2. Compiler Optimization with ROBUST FALLBACK
        # The crash happened here because Windows requires MSVC (cl.exe)
        if config.compile_model and hasattr(torch, 'compile'):
            
            # Check for Windows C++ Compiler
            is_windows = platform.system() == 'Windows'
            has_compiler = shutil.which('cl') is not None
            
            if is_windows and not has_compiler:
                self.logger.warning("⚠️ Windows detected without C++ Compiler (cl.exe).")
                self.logger.warning("   Disabling torch.compile() to prevent crash.")
                self.logger.info("   (Install Visual Studio Build Tools to enable compilation speedups)")
                config.compile_model = False
            else:
                self.logger.info("🚀 Compiling model with torch.compile() for speed...")
                try:
                    self.model = torch.compile(self.model)
                except Exception as e:
                    self.logger.warning(f"Compilation failed during init: {e}")
                    config.compile_model = False

        # 3. Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)
        
        # Metrics
        self.loss_history = deque(maxlen=config.evaluation_frequency)
        self.step_count = 0
        
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('AdaptiveFramework')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            logger.addHandler(handler)
        return logger
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for inference."""
        x = x.to(self.device)
        # Inference mode is more efficient
        with torch.inference_mode():
            output, log_var = self.model(x, return_internals=False)
        return output, log_var
    
    def train_step(self, 
                   input_data: torch.Tensor,
                   target: torch.Tensor) -> Dict[str, float]:
        """
        Single training step with introspection and adaptation.
        """
        self.model.train()
        input_data = input_data.to(self.device)
        target = target.to(self.device)
        
        self.optimizer.zero_grad(set_to_none=True) # Perf optimization
        
        # 1. Forward pass
        output, log_var, internals = self.model(input_data, return_internals=True)
        
        # 2. Robust Loss Calculation
        # FIX: Clamp log_var to prevent -inf/inf instability
        # -10.0 corresponds to sigma ~ 0.006 (high certainty)
        # +10.0 corresponds to sigma ~ 148.0 (high uncertainty)
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        
        precision = torch.exp(-log_var)
        mse = (output - target) ** 2
        
        # Gaussian NLL Loss: 0.5 * (log_var + (y-pred)^2 / var)
        loss = torch.mean(0.5 * (log_var + mse * precision))
        #==============================EWC memory 
        if self.ewc.is_enabled():
            ewc_loss = self.ewc.compute_penalty()
            loss += ewc_loss
        
        # 3. NaN Guard
        if torch.isnan(loss) or torch.isinf(loss):
            self.logger.warning(f"⚠️ NaN Loss detected at step {self.step_count}. Skipping update.")
            return {'loss': 10.0, 'uncertainty_mean': 0.0} # Return dummy high loss
        
        # 4. Backward & Step
        loss.backward()
        
        # Gradient clipping is crucial for Transformer stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # 5. Metrics & History
        loss_val = loss.item()
        metrics = {
            'loss': loss_val,
            'uncertainty_mean': log_var.mean().item()
        }
        
        self.loss_history.append(loss_val)
        
        # 6. Adaptation Trigger
        if self.step_count % self.config.evaluation_frequency == 0:
            if len(self.loss_history) > 10:
                current_loss = np.mean(list(self.loss_history)[-10:])
                previous_loss = np.mean(list(self.loss_history)[-20:-10]) if len(self.loss_history) >= 20 else current_loss + 0.1
                
                weight_adapt_mag, _ = self.monitor.adapt_weights(
                    current_loss, previous_loss, internals
                )
                metrics['weight_adaptation_magnitude'] = weight_adapt_mag
        
        self.step_count += 1
        return metrics
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Handle compiled models (unwrap prefix)
        state_dict = self.model.state_dict()
        if hasattr(self.model, '_orig_mod'):
             state_dict = self.model._orig_mod.state_dict()
             
        torch.save({
            'model_state': state_dict,
            'config': self.config,
            'step_count': self.step_count
        }, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Handle loading into compiled model
        if hasattr(self.model, '_orig_mod'):
            self.model._orig_mod.load_state_dict(checkpoint['model_state'])
        else:
            self.model.load_state_dict(checkpoint['model_state'])
            
        self.step_count = checkpoint.get('step_count', 0)
        self.logger.info(f"Checkpoint loaded from {path}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get summary of metrics"""
        if not self.loss_history:
            return {}
        return {'avg_recent_loss': np.mean(self.loss_history)}
    
