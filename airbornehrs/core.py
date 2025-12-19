"""
Core Adaptive Meta-Learning Framework (Universal V6.1 - "Still" Edition)
========================================================================
The Universal Wrapper that turns ANY PyTorch model into a Self-Learning System.

INTEGRATION FIXES (V6.1.1):
1. Reptile Integration: Connects MetaController for "Fast/Slow" weight syncing.
2. RL Stabilization: Uses Z-Score clamping to prevent reward explosion during domain shifts.
3. Full Circle: Closes the loop between Introspection (RL) and Optimization (Reptile).
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
import sys
import platform
import shutil
from datetime import datetime

# Import EWC and Meta-Controller

from .ewc import EWCHandler
from .meta_controller import MetaController, MetaControllerConfig

# OPTIMIZATION: Use Tensor Cores on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

# ==================== CONFIGURATION ====================

@dataclass
class AdaptiveFrameworkConfig:
    """
    Configuration for the Universal Framework.
    """
    # Architecture (for IntrospectionModule)
    model_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1
    
    # Learning parameters
    learning_rate: float = 1e-3
    meta_learning_rate: float = 1e-4
    
    # Plasticity: How much the model can 'edit' itself directly
    weight_adaptation_lr: float = 1e-5 
    bias_adaptation_lr: float = 1e-5
    adaptation_threshold: float = 0.05
    
    # Introspection
    telemetry_dim: int = 4 # Mean, Var, Max, Min
    
    # Memory
    feedback_buffer_size: int = 10000
    evaluation_frequency: int = 10
    
    # Optimization
    compile_model: bool = True 
    use_amp: bool = False
    
    # Device (auto-detected if None)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging
    log_frequency: int = 50
    checkpoint_frequency: int = 500
    @classmethod
    def quick_start(cls):
        """Beginner-friendly CPU config"""
        return cls(
            model_dim=128, 
            num_layers=4, 
            device='cpu', 
            compile_model=False,
            learning_rate=1e-3
        )

    @classmethod
    def production(cls):
        """High-performance GPU config"""
        return cls(
            model_dim=512, 
            num_layers=12, 
            device='cuda', 
            use_amp=True,
            compile_model=True
        )


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


# ==================== UNIVERSAL COMPONENTS ====================

class FeedbackBuffer:
    """Robust Experience Replay Buffer using Reservoir Sampling."""
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
            replace_idx = random.randint(0, self.total_seen)
            if replace_idx < self.capacity:
                self.buffer[replace_idx] = snapshot
        self.total_seen += 1


class IntrospectionEngine(nn.Module):
    """
    The 'Meta-Brain' (V6.1 Policy Network).
    Outputs a DISTRIBUTION of Affine Modifiers to enable REINFORCE training.
    """
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        
        # 1. State Monitor (Consciousness/Uncertainty)
        self.state_monitor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1) # Output: Log Variance
        )
        
        # 2. Hyper-Policy (Outputs Mu and Sigma for Modifiers)
        # We output 4 params: Scale_Mu, Scale_Sigma, Shift_Mu, Shift_Sigma
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4) 
        )
        
    def forward(self, global_state):
        # Predict Uncertainty (Differentiable, linked to Loss)
        log_var = self.state_monitor(global_state)
        
        # Predict Policy parameters
        policy_out = self.policy_net(global_state)
        
        # Split into Mu and Log-Sigma (using log for numerical stability)
        # Shape: [Batch, 4] -> [Batch, 2], [Batch, 2]
        mu, log_sigma = policy_out.chunk(2, dim=-1)
        sigma = torch.exp(torch.clamp(log_sigma, -5, 2)) # Clamp sigma
        
        # Create Distribution
        dist = torch.distributions.Normal(mu, sigma)
        
        # Sample Action (Affine Modifiers)
        # We sample to explore the "Action Space" of weight editing
        action = dist.sample()
        
        # Calculate Log Prob (Critical for REINFORCE)
        # Sum over the 2 dimensions (Scale & Shift) to get prob of the vector
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return log_var, action, log_prob


class PerformanceMonitor:
    """
    The 'Cortex' that governs adaptation.
    Executes the Affine Transformations commanded by the IntrospectionEngine.
    """
    def __init__(self, model: nn.Module, config: AdaptiveFrameworkConfig, device):
        self.model = model
        self.config = config
        self.device = device

    def adapt_weights(self, 
                      current_loss: float, 
                      previous_loss: float,
                      activations: Dict[str, Any]) -> float:
        
        affine_modifiers = activations.get('affine_modifiers', None)
        # We now pass the buffer AND the map
        telemetry_buffer = activations.get('telemetry_buffer', None) 
        layer_map = activations.get('layer_map', {}) 
        
        if affine_modifiers is None: return 0.0
        
        # Decode Intent
        if affine_modifiers.ndim > 1: affine_modifiers = affine_modifiers.mean(dim=0)
        raw_scale, raw_shift = affine_modifiers[0], affine_modifiers[1]
        
        # Early exit if effect is negligible to save compute
        if abs(raw_scale) < 1e-4 and abs(raw_shift) < 1e-4:
            return 0.0

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param_importance = 0.1
                    
                    # FIND THE LAYER INDEX FOR THIS PARAMETER
                    for layer_name, idx in layer_map.items():
                        if layer_name in name:
                            # Direct read from buffer row [idx]
                            # Column 0 = Mean, Column 1 = Var
                            stats = telemetry_buffer[idx]
                            mean_act = stats[0].abs()
                            var_act = stats[1]
                            
                            # Re-calculate importance on the fly
                            param_importance = (mean_act * var_act).item()
                            break
                    
                    # Apply updates
                    scale_factor = raw_scale * self.config.weight_adaptation_lr * param_importance
                    shift_factor = raw_shift * self.config.weight_adaptation_lr * param_importance
                    
                    if param.ndim == 1:
                        param.mul_(1.0 + scale_factor)
                        param.add_(shift_factor)
                    elif param.ndim >= 2:
                        param.mul_(1.0 + scale_factor)

        return (abs(raw_scale) + abs(raw_shift)).item()


# ==================== UNIVERSAL FRAMEWORK ====================

class AdaptiveFramework(nn.Module):
    """
    The Universal Wrapper (V6.1).
    Pass ANY PyTorch model here, and it becomes self-learning.
    """
    
    def __init__(self, user_model: nn.Module, config: AdaptiveFrameworkConfig = None, device=None):
        super().__init__()
        
        if config is None: config = AdaptiveFrameworkConfig()
        if device is None: device = torch.device(config.device)
             
        self.config = config
        self.device = device
        self.logger = self._setup_logging()
        
        self.model = user_model.to(self.device)
        self._attach_hooks()
        
        # 1. The "Mind" (RL Policy)
        self.introspection_engine = IntrospectionEngine(
            input_dim=config.telemetry_dim
        ).to(self.device)
        
        # 2. The "Cortex" (Weight Editor)
        self.monitor = PerformanceMonitor(self.model, config, self.device)
        self.feedback_buffer = FeedbackBuffer(config, self.device)
        self.ewc = EWCHandler(self.model, ewc_lambda=0.4)
        
       
        
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)
         # 3. The "Meta-Controller" (Reptile Optimizer)
        # FIX A: Initialize MetaController for Reptile integration
        self.meta_controller = MetaController(self, MetaControllerConfig(
            use_reptile=True,
            reptile_update_interval=5
        ))
        
        self.meta_optimizer = AdamW(self.introspection_engine.parameters(), 
                                   lr=config.meta_learning_rate,
                                   weight_decay=1e-2) 
        
        self.loss_history = deque(maxlen=100) # Increased for Z-Score calc
        self.meta_log_probs = []
        self.step_count = 0
        
        # RL Initialization
        self.reward_baseline = 0.0
        self.alpha = 0.1
        
        # Compilation
        if config.compile_model and hasattr(torch, 'compile'):
            is_windows = platform.system() == 'Windows'
            has_cl = shutil.which('cl') is not None
            
            if is_windows and not has_cl:
                self.logger.warning("⚠️ Windows detected without C++ Compiler. Disabling torch.compile.")
            else:
                try:
                    self.logger.info("🚀 Compiling model for speed...")
                    self.model = torch.compile(self.model)
                except Exception as e:
                    self.logger.warning(f"Compilation failed: {e}")

    def _setup_logging(self):
        logger = logging.getLogger('AdaptiveFramework')
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _attach_hooks(self):
        valid_types = (nn.Linear, nn.Conv2d, nn.Conv1d, nn.LSTM, nn.GRU, nn.MultiheadAttention)
        self.layer_map = {}
        idx = 0
        for name, module in self.model.named_modules():
            if isinstance(module, valid_types):
                self.layer_map[name] = idx
                module.register_forward_hook(self._generate_fast_hook(idx))
                idx += 1
        
        self.num_tracked_layers = idx
        self.telemetry_buffer = torch.zeros(
            (self.num_tracked_layers, 4), 
            device=self.device, 
            dtype=torch.float32,
            requires_grad=False
        )
        self.logger.info(f"⚡ Fast Telemetry Bus established for {idx} layers.")

    def _generate_fast_hook(self, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple): activation = output[0]
            elif isinstance(output, dict): activation = list(output.values())[0]
            else: activation = output
            
            act_flat = activation.detach().flatten()
            if act_flat.numel() > 0:
                self.telemetry_buffer[layer_idx, 0] = act_flat.mean()
                self.telemetry_buffer[layer_idx, 1] = act_flat.var(unbiased=False)
                self.telemetry_buffer[layer_idx, 2] = act_flat.max()
                self.telemetry_buffer[layer_idx, 3] = act_flat.min()
        return hook

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        
        log_var = torch.tensor(0.0).to(self.device)
        affine_modifiers = None
        
        try:
            # FAST PATH: No stacking needed. The buffer IS the state.
            global_state = self.telemetry_buffer.mean(dim=0)
            
            # RL Step
            log_var, action, log_prob = self.introspection_engine(global_state)
            
            affine_modifiers = action.detach()
            self.meta_log_probs.append(log_prob)
                
        except Exception as e:
            self.logger.warning(f"Introspection failed: {e}")
            log_var = torch.tensor(0.0).to(self.device)
            affine_modifiers = None
            self.meta_log_probs.clear()
            
        return output, log_var, affine_modifiers

    def train_step(self, input_data, target_data):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self.meta_optimizer.zero_grad(set_to_none=True)
        
        # 1. Forward Pass
        output, log_var, affine_modifiers = self.forward(input_data)
        
        # Handle Output Types
        pred = output
        if hasattr(output, 'logits'): pred = output.logits
        elif isinstance(output, tuple): pred = output[0]
        
        # 2. Loss Calculation
        if pred.shape == target_data.shape:
            mse = (pred - target_data) ** 2
            precision = torch.exp(-log_var)
            loss = torch.mean(0.5 * (log_var + mse * precision))
        elif pred.dim() > target_data.dim(): 
             ce_loss = F.cross_entropy(pred.view(-1, pred.size(-1)), target_data.view(-1), reduction='none')
             precision = torch.exp(-log_var)
             loss = torch.mean(0.5 * (log_var + ce_loss * precision))
        else:
             loss = F.mse_loss(pred.float(), target_data.float())
             
        # 3. NaN Guard (Early Exit)
        if torch.isnan(loss) or torch.isinf(loss):
             if len(self.meta_log_probs) > 0: self.meta_log_probs.pop()
             return {'loss': 10.0, 'uncertainty': 0.0, 'weight_adaptation': 0.0}

        # ==========================================================
        # SOTA FIX: SURPRISE-BASED CONSOLIDATION (Auto-EWC)
        # ==========================================================
        current_loss_val = loss.item()
        trigger_ewc = False
        z_score = 0.0

        # Only check for surprise if we have enough history to form a baseline
        if len(self.loss_history) > 20:
            hist_mean = np.mean(self.loss_history)
            hist_std = np.std(self.loss_history) + 1e-9
            z_score = (current_loss_val - hist_mean) / hist_std
            
            # CRITICAL: If Z-Score > 3.0, we are in a new domain.
            if z_score > 3.0:
                # Add a cooldown (e.g., 50 steps) so we don't re-lock constantly during a spike
                last_consolidation = getattr(self, 'last_consolidation_step', 0)
                if self.step_count > last_consolidation + 50:
                    trigger_ewc = True
                    self.last_consolidation_step = self.step_count

        if trigger_ewc:
            # Lock memories from the BUFFER (representing the previous task)
            # before we learn from this new "surprising" data.
            self.ewc.consolidate_from_buffer(self.feedback_buffer)
            
            # Clear history so the new high loss becomes the new normal baseline
            self.loss_history.clear() 

        # ==========================================================

        # 4. Backward & Step (Main Model)
        
        # Apply EWC Penalty (if enabled by previous trigger or manually)
        if self.ewc.is_enabled():
            loss += self.ewc.compute_penalty()
            
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # 5. RL Update (Introspection Engine)
        
        # Init baseline if empty
        if self.reward_baseline == 0.0:
            self.reward_baseline = current_loss_val
            
        # Calculate Advantage
        advantage = self.reward_baseline - current_loss_val
        
        # Z-Score Clamping for RL (Prevent exploding gradients on surprise)
        if abs(z_score) > 3.0:
             advantage = torch.clamp(torch.tensor(advantage), min=-1.0, max=1.0).item()
        
        # Update Baseline
        self.reward_baseline = (1 - self.alpha) * self.reward_baseline + self.alpha * current_loss_val
        
        if len(self.meta_log_probs) > 0:
            scaled_reward = advantage * 10.0 
            log_prob = self.meta_log_probs[-1]
            policy_loss = -log_prob * scaled_reward
            policy_loss.backward()
            self.meta_optimizer.step()
            self.meta_log_probs.clear()
        
        # 6. Meta-Controller Adaptation (Reptile)
        # This syncs the "Fast Weights" (current step) with "Slow Weights" (Anchor)
        self.meta_controller.adapt(loss=current_loss_val)
        
        # 7. Explicit Weight Editing (Direct Cortex Edit)
        weight_adapt_mag = 0.0
        if self.step_count % self.config.evaluation_frequency == 0:
            avg_loss = np.mean(self.loss_history) if self.loss_history else loss.item()
            internals = {
                'affine_modifiers': affine_modifiers, 
                'telemetry_buffer': self.telemetry_buffer,
                'layer_map': self.layer_map
            }
            weight_adapt_mag = self.monitor.adapt_weights(
                current_loss=loss.item(),
                previous_loss=avg_loss,
                activations=internals
            )
            
        # 8. Checkpoints & Housekeeping
        if self.step_count % self.config.checkpoint_frequency == 0:
            self.save_checkpoint(f"checkpoints/step_{self.step_count}.pt")

        self.loss_history.append(loss.item())
        self.feedback_buffer.add(input_data, pred, target_data, -loss.item(), loss.item())
        
        # AUTOMATIC DREAMING (Experience Replay)
        # Every 10 steps, replay 1 epoch of memories to lock them in.
        replay_loss = 0.0
        if self.step_count > 0 and self.step_count % 10 == 0:
             dream_metrics = self.learn_from_buffer(batch_size=16, num_epochs=1)
             replay_loss = dream_metrics.get('replay_loss', 0.0)
        self.step_count += 1
        
        return {
            "loss": loss.item(),
            "replay_loss": replay_loss,
            "uncertainty_mean": log_var.item(),
            "weight_adaptation": weight_adapt_mag,
            "surprise_z_score": float(z_score) if isinstance(z_score, (float, int)) else z_score.item()
        }
    
    def learn_from_buffer(self, batch_size: int = 32, num_epochs: int = 1) -> Dict[str, float]:
        """
        Active Replay ("Dreaming"): Re-trains on past experiences to consolidate memory.
        
        This closes the loop by:
        1. Sampling diverse memories from the FeedbackBuffer.
        2. Re-running them through the full optimization cycle (Meta-Controller + EWC).
        3. syncing the "Fast Weights" with long-term storage.
        
        Args:
            batch_size (int): Number of memories to replay per step.
            num_epochs (int): How many passes to make over the sampled memories.
            
        Returns:
            Dict containing average loss during replay.
        """
        # 1. Safety Check: Do we have enough memories?
        if len(self.feedback_buffer.buffer) < batch_size:
            # If buffer is small, just learn from what we have
            if len(self.feedback_buffer.buffer) < 10:
                return {} # Too few to be useful
            batch_size = len(self.feedback_buffer.buffer)

        self.logger.info(f"💤 Dreaming: Consolidating {num_epochs} epochs from {len(self.feedback_buffer.buffer)} memories...")
        
        replay_losses = []
        
        # 2. Optimization Loop
        self.model.train()
        
        for epoch in range(num_epochs):
            # Reservoir Sampling: Randomly pick experiences
            # We use random.sample for diversity (breaking temporal correlations)
            samples = random.sample(self.feedback_buffer.buffer, batch_size)
            
            # 3. Reconstruct Tensors & Move to Device
            # We must detach to ensure we don't backprop into the buffer history
            inputs = torch.stack([s.input_data for s in samples]).to(self.device)
            targets = torch.stack([s.target for s in samples]).to(self.device)
            
            # 4. The Sync Step: Call train_step()
            # This is CRITICAL. By calling train_step instead of doing a manual backward pass,
            # we force the replay to go through:
            # - The Introspection Engine (is this memory surprising?)
            # - The EWC Handler (does this violate previous constraints?)
            # - The Meta-Controller (should we adjust LR?)
            metrics = self.train_step(inputs, targets)
            replay_losses.append(metrics['loss'])
            
        avg_loss = np.mean(replay_losses) if replay_losses else 0.0
        self.logger.info(f"✨ Dream Complete. Avg Replay Loss: {avg_loss:.4f}")
        
        return {'replay_loss': avg_loss}

    def save_checkpoint(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state_dict = self.model.state_dict()
        if hasattr(self.model, '_orig_mod'): state_dict = self.model._orig_mod.state_dict()
             
        ewc_state = self.ewc.fisher_dict if hasattr(self.ewc, 'fisher_dict') else None
        
        torch.save({
            'model_state': state_dict,
            'introspection_engine': self.introspection_engine.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'meta_optimizer': self.meta_optimizer.state_dict(),
            'ewc_fisher': ewc_state, 
            'config': self.config,
            'step_count': self.step_count
        }, path)
        self.logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if hasattr(self.model, '_orig_mod'): self.model._orig_mod.load_state_dict(ckpt['model_state'])
        else: self.model.load_state_dict(ckpt['model_state'])
            
        if 'introspection_engine' in ckpt:
            self.introspection_engine.load_state_dict(ckpt['introspection_engine'])
        
        if 'ewc_fisher' in ckpt and ckpt['ewc_fisher'] is not None:
            self.ewc.fisher_dict = ckpt['ewc_fisher']
            
        if 'meta_optimizer' in ckpt:
            self.meta_optimizer.load_state_dict(ckpt['meta_optimizer'])
            
        self.step_count = ckpt.get('step_count', 0)
        self.logger.info(f"Checkpoint loaded from {path}")

    def get_metrics(self) -> Dict[str, Any]:
        return {
            'avg_recent_loss': np.mean(self.loss_history) if self.loss_history else 0.0,
            'step_count': self.step_count
        }

    def consolidate_memory(self, data_loader):
        if data_loader is not None:
             self.logger.info("🧠 Consolidating Memories (Full EWC Scan)...")
             self.ewc.save_task_memory(data_loader)
             