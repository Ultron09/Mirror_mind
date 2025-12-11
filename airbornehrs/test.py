"""
Core Adaptive Meta-Learning Framework (Dynamic V3.0)
====================================================
The central engine for MirrorMind. Implements:
1. Dynamic Topology: Runtime layer growth (Mitosis)
2. Hyper-Plasticity: Neural Weight Editing via Hypernetworks
3. Deep Introspection: Per-layer telemetry and variance monitoring

This module transforms standard PyTorch models into "Liquid" architectures
that can grow and adapt their structure based on learning stress.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional, Any, Union
import numpy as np
import random
from collections import deque
from pathlib import Path
import logging
from datetime import datetime
import sys
import platform
import shutil
import copy
import math

# Try to import EWC, handle if missing for standalone testing
try:
    from airbornehrs.ewc import EWCHandler
except ImportError:
    EWCHandler = None

# OPTIMIZATION: Use Tensor Cores on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

# ==================== CONFIGURATION ====================

@dataclass
class AdaptiveFrameworkConfig:
    """
    Configuration for the Dynamic Adaptive Framework.
    """
    # --- Model Architecture (Initial Seed) ---
    model_dim: int = 256
    num_layers: int = 6      # Starting depth
    num_heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1
    
    # --- Dynamic Architecture Limits (The Safety Leash) ---
    max_depth: int = 24             # Absolute hard limit on growth
    min_depth: int = 2              # Minimum functional depth
    growth_threshold: float = 2.0   # Loss must exceed this to trigger Mitosis
    growth_cooldown: int = 500      # Steps to wait between growth events
    
    # --- Optimization ---
    compile_model: bool = False     # Must be False for dynamic graphs
    use_amp: bool = False           # Mixed Precision Training
    learning_rate: float = 1e-4
    
    # --- Hyperplasticity (Weight Editing) ---
    weight_adaptation_lr: float = 1e-5  # Magnitude of subconscious edits
    adaptation_threshold: float = 0.05  # Minimum loss improvement to skip editing
    
    # --- Memory & Replay ---
    feedback_buffer_size: int = 10000
    evaluation_frequency: int = 50
    checkpoint_frequency: int = 1000

# ==================== DATA STRUCTURES ====================

@dataclass
class PerformanceSnapshot:
    """Standard container for experience replay"""
    input_data: torch.Tensor
    output: torch.Tensor
    target: torch.Tensor
    reward: float
    loss: float
    timestamp: float
    episode: int
    
    def to_device(self, device):
        self.input_data = self.input_data.to(device)
        self.output = self.output.to(device)
        self.target = self.target.to(device)
        return self

# ==================== WHITEBOX MONITORING ====================

class LayerMonitor(nn.Module):
    """
    Transparent wrapper for any neural block.
    Acts as the 'Nervous System', capturing telemetry without altering data flow.
    """
    def __init__(self, layer: nn.Module, index: int):
        super().__init__()
        self.layer = layer
        self.index = index
        self.stats = {} # Transient storage for the current forward pass

    def forward(self, x, return_stats=False):
        # 1. Execute the wrapped layer
        out = self.layer(x)
        
        # 2. Capture Telemetry (Introspection)
        if return_stats:
            with torch.no_grad():
                # Activity: How "loud" is this layer shouting?
                activity = out.abs().mean().item()
                
                # Variance: How complex represents the information processing?
                # Low variance = Dead layer / Pass-through
                # High variance = Heavy processing / Confusion
                variance = out.var().item()
                
                self.stats = {
                    'index': self.index,
                    'activity': activity,
                    'var': variance,
                    'shape': tuple(out.shape)
                }
        return out

    def __repr__(self):
        return f"LayerMonitor(idx={self.index}, wrapped={type(self.layer).__name__})"

# ==================== DYNAMIC CONTAINER ====================

class DynamicTransformer(nn.Module):
    """
    A growable, mutable Transformer architecture.
    Replaces static nn.ModuleList with dynamic management logic.
    """
    def __init__(self, config: AdaptiveFrameworkConfig):
        super().__init__()
        self.config = config
        
        # --- Static Interfaces (Sensors & Actuators) ---
        self.embedding = nn.Linear(config.model_dim, config.model_dim)
        self.pos_encoder = nn.Linear(1, config.model_dim) # Simple continuous embedding
        
        self.head = nn.Linear(config.model_dim, config.model_dim)
        self.uncertainty = nn.Linear(config.model_dim, 1)
        
        # --- The Hypernetwork (Subconscious Weight Editor) ---
        # This network predicts how to fix the weights of the main body
        self.weight_editor = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim),
            nn.Tanh(),
            nn.Linear(config.model_dim, config.model_dim)
        )
        
        # --- Dynamic Body (The Liquid Layers) ---
        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            self._add_layer_block(i)
            
    def _add_layer_block(self, index: int, source_block: nn.Module = None):
        """
        Internal factory: Creates or Clones a layer block.
        """
        if source_block:
            # MITOSIS PATH: Cloning an existing layer
            # Deep copy to ensure independence
            new_block = copy.deepcopy(source_block)
            
            # MUTATION: Add Gaussian noise to break symmetry
            # This is critical. Exact copies perform no new computation initially.
            with torch.no_grad():
                for p in new_block.parameters():
                    if p.requires_grad:
                        noise = torch.randn_like(p) * 0.01  # 1% Mutation
                        p.add_(noise)
        else:
            # GENESIS PATH: Fresh initialization
            new_block = nn.TransformerEncoderLayer(
                d_model=self.config.model_dim,
                nhead=self.config.num_heads,
                dim_feedforward=self.config.ff_dim,
                dropout=self.config.dropout,
                batch_first=True,
                norm_first=True, # Pre-LN is essential for deep/growing nets
                activation="gelu"
            )
            
        # Wrap in Monitor
        probe = LayerMonitor(new_block, index=index)
        
        # Logic to Insert vs Append
        if index < len(self.layers):
            self.layers.insert(index, probe)
        else:
            self.layers.append(probe)
            
        # Re-index all probes to keep telemetry clean
        for i, p in enumerate(self.layers):
            p.index = i
            
        return probe

    def grow(self, target_index: int = -1):
        """
        Public API to trigger architecture growth (Mitosis).
        
        Args:
            target_index: Where to insert the new layer. -1 = End.
        
        Returns:
            The new LayerMonitor object (so it can be registered with optimizer).
        """
        if len(self.layers) >= self.config.max_depth:
            return None
            
        # Determine insertion point
        idx = len(self.layers) if target_index == -1 else target_index
        
        # Select parent for cloning (Structural Inheritance)
        # If growing at 0, we have no parent, so use fresh init (None)
        parent = self.layers[idx-1].layer if idx > 0 else None
        
        # Execute Growth
        new_probe = self._add_layer_block(idx, source_block=parent)
        return new_probe

    def forward(self, x, return_telemetry=False):
        # 1. Input Embedding
        x = self.embedding(x)
        
        telemetry_stream = []
        
        # 2. Dynamic Flow (Iterate through current topology)
        for probe in self.layers:
            x = probe(x, return_stats=return_telemetry)
            if return_telemetry:
                telemetry_stream.append(probe.stats)
        
        # 3. Introspection (Hypernetwork Pass)
        # Summarize the entire thought process into a global state vector
        global_state = x.mean(dim=1)
        
        # Generate the Correction Vector (The "Subconscious" urge)
        # This vector represents how the weights *should* shift to fix errors
        correction_vector = self.weight_editor(global_state)
        
        # 4. Outputs
        logits = self.head(x)
        log_var = self.uncertainty(x)
        
        if return_telemetry:
            return logits, log_var, {
                'telemetry': telemetry_stream,
                'correction_vector': correction_vector
            }
        return logits, log_var

# ==================== PERFORMANCE MONITOR ====================

class PerformanceMonitor:
    """
    The 'Cortex' that governs adaptation.
    Decides WHEN to edit weights and WHEN to grow the brain.
    """
    def __init__(self, model: DynamicTransformer, config: AdaptiveFrameworkConfig, device):
        self.model = model
        self.config = config
        self.device = device
        self.last_growth_step = 0
        
    def adapt_weights(self, current_loss: float, prev_loss: float, internals: Dict):
        """
        Applies Hyperplasticity (Neural Weight Editing).
        Uses the correction vector from the Hypernetwork to nudge biases.
        """
        if 'correction_vector' not in internals:
            return 0.0
            
        # Heuristic: If we are improving fast enough, don't mess with weights.
        improvement = prev_loss - current_loss
        if abs(improvement) > self.config.adaptation_threshold:
            return 0.0
            
        # Consensus Correction: Average the batch's intent
        correction = internals['correction_vector'].mean(dim=0)
        magnitude = 0.0
        
        # Apply edits to 1D parameters (Biases/LayerNorms)
        # This changes the "Activation Thresholds" without destroying learned patterns.
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and 'layers' in name:
                    # Check dimension match
                    if param.ndim == 1 and param.shape[0] == correction.shape[0]:
                        
                        # Delta = LearningRate * Correction
                        delta = self.config.weight_adaptation_lr * correction
                        
                        # Apply
                        param.add_(delta)
                        magnitude += delta.abs().mean().item()
                        
        return magnitude

    def check_growth_triggers(self, step_count: int, loss: float, telemetry: List[Dict]) -> int:
        """
        Evaluates metrics to decide if Mitosis is needed.
        Returns: Index to grow at, or -1 if no growth.
        """
        # 1. Cooldown Check
        if (step_count - self.last_growth_step) < self.config.growth_cooldown:
            return -1
            
        # 2. Stress Check (Loss Threshold)
        if loss < self.config.growth_threshold:
            return -1
            
        # 3. Localization Strategy (Where is the bottleneck?)
        # We look for the layer with the highest Variance (Confusion/Activity)
        if not telemetry:
            return -1
            
        max_var = 0.0
        target_idx = -1
        
        for layer_stat in telemetry:
            if layer_stat['var'] > max_var:
                max_var = layer_stat['var']
                target_idx = layer_stat['index']
        
        # Only grow if the bottleneck is significant
        if max_var > 1.0: # Threshold for high variance
            self.last_growth_step = step_count
            return target_idx + 1 # Grow *after* the bottleneck to process it
            
        return -1

# ==================== FEEDBACK BUFFER ====================

class FeedbackBuffer:
    """Reservoir Sampling for Long-Term Memory"""
    def __init__(self, config: AdaptiveFrameworkConfig, device):
        self.capacity = config.feedback_buffer_size
        self.device = device
        self.buffer: List[PerformanceSnapshot] = []
        self.total_seen = 0

    def add(self, input_data, output, target, reward, loss):
        snapshot = PerformanceSnapshot(
            input_data=input_data.detach().cpu(),
            output=output.detach().cpu(),
            target=target.detach().cpu(),
            reward=reward,
            loss=loss,
            timestamp=datetime.now().timestamp(),
            episode=self.total_seen
        )
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(snapshot)
        else:
            # Algorithm R
            replace_idx = random.randint(0, self.total_seen)
            if replace_idx < self.capacity:
                self.buffer[replace_idx] = snapshot
        self.total_seen += 1

# ==================== MAIN FRAMEWORK ====================

class AdaptiveFramework:
    """
    The Orchestrator. Manages the lifecycle of the Dynamic Model.
    Handles the Training Loop, Optimizer Sync, and Evolution events.
    """
    def __init__(self, config: AdaptiveFrameworkConfig, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.logger = self._setup_logging()
        
        # 1. Initialize Dynamic Model
        self.model = DynamicTransformer(config).to(self.device)
        
        # 2. Initialize Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)
        
        # 3. Subsystems
        self.monitor = PerformanceMonitor(self.model, config, self.device)
        self.feedback_buffer = FeedbackBuffer(config, self.device)
        self.ewc = EWCHandler(self.model) if EWCHandler else None
        
        # 4. State Tracking
        self.step_count = 0
        self.loss_history = deque(maxlen=config.evaluation_frequency)
        
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('AdaptiveFramework')
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        with torch.inference_mode():
            logits, log_var = self.model(x, return_telemetry=False)
        return logits, log_var

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """
        The Learning Cycle:
        Forward -> Introspect -> Evolve (Maybe) -> Learn -> Adapt Weights
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        x, y = x.to(self.device), y.to(self.device)
        
        # 1. Forward with Telemetry
        # This captures the internal state of every layer before we update
        logits, log_var, internals = self.model(x, return_telemetry=True)
        
        # 2. Robust Loss Calculation
        # Clamp uncertainty to prevent numerical instability
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        precision = torch.exp(-log_var)
        mse = (logits - y) ** 2
        
        # NLL Loss = 0.5 * (log(var) + (error^2 / var))
        loss = torch.mean(0.5 * (log_var + mse * precision))
        
        # Add EWC Penalty if active (Memory Protection)
        if self.ewc and self.ewc.is_enabled():
            loss += self.ewc.compute_penalty()
            
        # NaN Guard
        if torch.isnan(loss) or torch.isinf(loss):
            self.logger.error(f"⚠️ NaN Loss at step {self.step_count}. Skipping update.")
            return {'loss': 10.0}
            
        # 3. Backward Pass
        loss.backward()
        
        # Gradient Clipping (Essential for dynamic depth)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # 4. Optimizer Step
        self.optimizer.step()
        
        # 5. POST-UPDATE: Evolution & Adaptation
        loss_val = loss.item()
        self.loss_history.append(loss_val)
        
        # A. Hyperplasticity (Weight Editing)
        # Run frequently to keep weights fluid
        if self.step_count % self.config.evaluation_frequency == 0:
            avg_loss = np.mean(self.loss_history)
            # Use avg loss as 'previous' reference
            mag = self.monitor.adapt_weights(loss_val, avg_loss + 0.1, internals)
            if mag > 0:
                pass # self.logger.debug(f"Weight Edit Magnitude: {mag:.5f}")
                
        # B. Architecture Growth (Mitosis)
        # Run sparingly to allow settlement
        if self.step_count % 100 == 0:
            growth_idx = self.monitor.check_growth_triggers(
                self.step_count, 
                loss_val, 
                internals['telemetry']
            )
            
            if growth_idx != -1:
                self._perform_mitosis(growth_idx)
                
        self.step_count += 1
        return {
            'loss': loss_val,
            'uncertainty_mean': log_var.mean().item(),
            'depth': len(self.model.layers)
        }

    def _perform_mitosis(self, index: int):
        """
        Executes the growth protocol and registers new parameters.
        """
        self.logger.info(f"🧬 EVOLUTION TRIGGERED: Growing layer at index {index}...")
        
        # 1. Grow the Model
        new_probe = self.model.grow(index)
        
        if new_probe:
            # 2. Move new layer to device
            new_probe.to(self.device)
            
            # 3. CRITICAL: Register with Optimizer
            # We must add the new parameters to the optimizer's tracking list
            # without resetting the momentum of existing parameters.
            self.optimizer.add_param_group({
                'params': new_probe.parameters(),
                'lr': self.config.learning_rate
            })
            
            self.logger.info(f"✅ GROWTH SUCCESS. New Depth: {len(self.model.layers)}")
        else:
            self.logger.warning("⚠️ GROWTH FAILED: Max depth reached or internal error.")

    def save_checkpoint(self, path: str):
        """
        Saves the dynamic model. We must save the config to know the structure.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config,
            'step_count': self.step_count,
            'layer_count': len(self.model.layers)
        }, path)
        self.logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """
        Loads a dynamic model.
        NOTE: Since topology changes, strict loading might fail if config differs.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        
        # 1. Rebuild Topology if needed
        saved_layers = ckpt.get('layer_count', self.config.num_layers)
        current_layers = len(self.model.layers)
        
        if saved_layers > current_layers:
            self.logger.info(f"Restoring topology: Growing from {current_layers} to {saved_layers} layers...")
            for i in range(current_layers, saved_layers):
                self.model.grow(-1) # Append to end to match count
                
        # 2. Load State
        try:
            self.model.load_state_dict(ckpt['model_state'])
            self.optimizer.load_state_dict(ckpt['optimizer_state'])
            self.step_count = ckpt['step_count']
            self.logger.info(f"Checkpoint loaded successfully.")
        except RuntimeError as e:
            self.logger.error(f"Topology mismatch during load: {e}")
            self.logger.warning("Attempting strict=False load (partial recovery)...")
            self.model.load_state_dict(ckpt['model_state'], strict=False)

    def get_metrics(self) -> Dict[str, Any]:
        return {
            'avg_recent_loss': np.mean(self.loss_history) if self.loss_history else 0.0,
            'network_depth': len(self.model.layers)
        }