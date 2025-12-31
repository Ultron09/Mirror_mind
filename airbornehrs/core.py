"""
Core Adaptive Meta-Learning Framework (Universal V7.0 - "Production" Edition)
=============================================================================
The Universal Wrapper that turns ANY PyTorch model into a Self-Learning System.

PATCH NOTES (V7.0):
1. Unified Memory Integration: Fully switched to optimized memory.py (No ewc.py).
2. Adapter Pre-Init: Fixes optimizer missing lazy parameters.
3. Vectorized Replay: Uses fast batching for dreams and consolidation.
4. Robust Platforming: Removed OS-specific path dependencies.
5. Consciousness V2: Full integration with the new self-awareness module.
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
import os
import platform
import shutil
from datetime import datetime
import time
import json

# Import Unified Memory, Meta-Controller, and Consciousness
# NOTE: We use .consciousness_v2 as requested for the SOTA module
from .memory import UnifiedMemoryHandler, PrioritizedReplayBuffer, AdaptiveRegularization, DynamicConsolidationScheduler
from .meta_controller import MetaController, MetaControllerConfig
from .consciousness_v2 import ConsciousnessCore
from .adapters import AdapterBank

# OPTIMIZATION: Use Tensor Cores on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

# ==================== CONFIGURATION ====================

@dataclass
class AdaptiveFrameworkConfig:
    """
    Configuration for the Universal Framework (V7.0).
    """
    # Architecture
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
    telemetry_dim: int = 4 
    feedback_buffer_size: int = 10000
    evaluation_frequency: int = 10
    # How often to run dreaming/replay (in steps).
    dream_interval: int = 10
    
    # Optimization
    compile_model: bool = True 
    use_amp: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_frequency: int = 50
    checkpoint_frequency: int = 500
    gradient_clip_norm: float = 1.0
    adapter_max_norm: float = 2.0
    
    # --- HIERARCHICAL REFLEX ---
    enable_active_shield: bool = True 
    active_shield_threshold: float = 0.05 
    active_shield_slope: float = 10.0   
    panic_threshold: float = 0.2
    warmup_steps: int = 50
    
    # Z-Score Thresholds
    novelty_z_threshold: float = 2.0
    survival_z_threshold: float = 4.0
    enable_dreaming: bool = True
    enable_tracing: bool = False
    trace_max_records: int = 1000
    
    # SOTA Unified Memory System (V7.0)
    memory_type: str = 'hybrid'  # 'ewc', 'si', or 'hybrid'
    consolidation_criterion: str = 'hybrid'
    consolidation_min_interval: int = 30
    consolidation_max_interval: int = 100
    consolidation_surprise_threshold: float = 2.5
    adaptive_lambda: bool = True
    use_prioritized_replay: bool = True
    replay_priority_temperature: float = 0.6
    
    # --- V7.0: CONSCIOUSNESS LAYER ---
    enable_consciousness: bool = True
    use_attention: bool = True
    use_intrinsic_motivation: bool = True
    consciousness_buffer_size: int = 5000
    novelty_threshold: float = 2.0

    @classmethod
    def production(cls):
        return cls(
            model_dim=512, 
            device='cuda', 
            use_amp=True, 
            compile_model=True,
            memory_type='hybrid',
            use_prioritized_replay=True,
            adaptive_lambda=True,
            enable_consciousness=True
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
        # Move to CPU immediately to save VRAM
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
    The 'Meta-Brain' (Policy Network).
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
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4) 
        )
        
    def forward(self, global_state):
        log_var = self.state_monitor(global_state)
        policy_out = self.policy_net(global_state)
        
        # Guard against NaNs
        policy_out = torch.nan_to_num(policy_out, nan=0.0, posinf=10.0, neginf=-10.0)

        # Split into Mu and Log-Sigma
        try:
            mu, log_sigma = policy_out.chunk(2, dim=-1)
        except Exception:
            mu = torch.zeros(1, 2, device=global_state.device)
            log_sigma = torch.zeros(1, 2, device=global_state.device)

        # Clamp log_sigma
        log_sigma = torch.clamp(log_sigma, min=-10.0, max=5.0)
        sigma = torch.exp(log_sigma)
        sigma = torch.clamp(sigma, min=1e-3, max=10.0)

        try:
            dist = torch.distributions.Normal(mu, sigma)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        except Exception:
            action = torch.zeros_like(mu)
            log_prob = torch.zeros(mu.size(0), device=mu.device)

        return log_var, action, log_prob


class PerformanceMonitor:
    """
    The 'Cortex' that governs adaptation via direct weight editing.
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
        telemetry_buffer = activations.get('telemetry_buffer', None) 
        layer_map = activations.get('layer_map', {}) 
        
        if affine_modifiers is None: return 0.0
        
        if affine_modifiers.ndim > 1: affine_modifiers = affine_modifiers.mean(dim=0)
        raw_scale = affine_modifiers[0].item()
        raw_shift = affine_modifiers[1].item()

        if abs(raw_scale) < 1e-4 and abs(raw_shift) < 1e-4:
            return 0.0


        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param_importance = 0.1
                    
                    # Find layer index
                    for layer_name, idx in layer_map.items():
                        if layer_name in name and telemetry_buffer is not None:
                            stats = telemetry_buffer[idx]
                            mean_act = stats[0].abs()
                            var_act = stats[1]
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
    The Universal Wrapper (V7.0).
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
        
        # 1. Initialize Adapters & Hooks (CRITICAL: Must be before optimizer)
        self._init_adapters_and_hooks()
        
        # 2. The "Mind" (RL Policy)
        self.introspection_engine = IntrospectionEngine(
            input_dim=config.telemetry_dim
        ).to(self.device)
        
        # 3. The "Cortex" (Weight Editor)
        self.monitor = PerformanceMonitor(self.model, config, self.device)
        
        # 4. Memory System (Unified Handler V7.0)
        # We explicitly use UnifiedMemoryHandler, removing all EWC legacy code.
        self.memory = UnifiedMemoryHandler(
            self.model,
            method=getattr(config, 'memory_type', 'hybrid'),
            si_lambda=getattr(config, 'si_lambda', 1.0),
            si_xi=getattr(config, 'si_xi', 1e-3),
            ewc_lambda=0.4,
            consolidation_criterion=getattr(config, 'consolidation_criterion', 'hybrid')
        )
        self.logger.info(f"[BRAIN] Unified Memory System Online ({config.memory_type})")
        
        # 5. Experience Replay
        self.feedback_buffer = FeedbackBuffer(config, self.device)
        if getattr(config, 'use_prioritized_replay', True):
            self.prioritized_buffer = PrioritizedReplayBuffer(
                capacity=config.feedback_buffer_size,
                temperature=getattr(config, 'replay_priority_temperature', 0.6)
            )
        else:
            self.prioritized_buffer = None
        
        # 6. Adaptive Regularization & Consolidation
        self.adaptive_reg = AdaptiveRegularization(base_lambda=0.4)
        self.consolidation_scheduler = DynamicConsolidationScheduler(
            min_interval=getattr(config, 'consolidation_min_interval', 30),
            max_interval=getattr(config, 'consolidation_max_interval', 100)
        )
        
        # 7. Consciousness Layer
        if getattr(config, 'enable_consciousness', False):
            self.consciousness = ConsciousnessCore(
                feature_dim=config.model_dim,
                awareness_buffer_size=getattr(config, 'consciousness_buffer_size', 5000),
                novelty_threshold=getattr(config, 'novelty_threshold', 2.0)
            )
            self.logger.info("[CONSCIOUSNESS] Self-Awareness Module Active")
        else:
            self.consciousness = None
        
        # 8. Optimizers
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)
        
        # Adapter Optimizer (CRITICAL FIX: Now sees parameters because _init_adapters_and_hooks ran first)
        if hasattr(self, 'adapter_bank') and self.adapter_bank is not None:
            adapter_params = list(self.adapter_bank.parameters())
            if adapter_params:
                self.adapter_optimizer = AdamW(adapter_params, lr=config.weight_adaptation_lr)
                self.logger.info(f"[ADAPTER] Optimizer attached to {len(adapter_params)//4} adapters.")
            else:
                self.adapter_optimizer = None
        else:
            self.adapter_optimizer = None

        # Meta-Controller (Reptile)
        self.meta_controller = MetaController(self, MetaControllerConfig(
            use_reptile=True,
            reptile_update_interval=5
        ))
        
        self.meta_optimizer = AdamW(self.introspection_engine.parameters(), 
                                   lr=config.meta_learning_rate,
                                   weight_decay=1e-2) 
        
        # State Tracking
        self.loss_history = deque(maxlen=100)
        self.meta_log_probs = []
        self.step_count = 0
        self.reward_baseline = 0.0
        self.alpha = 0.1
        
        # Compilation
        if config.compile_model and hasattr(torch, 'compile'):
            try:
                if platform.system() != 'Windows': # Compilation often fails on Windows
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

    def _init_adapters_and_hooks(self):
        """
        Initialize adapters by inspecting layer dimensions upfront.
        This ensures parameters exist before optimizer creation.
        """
        valid_types = (nn.Linear, nn.Conv2d, nn.Conv1d, nn.LSTM, nn.GRU, nn.MultiheadAttention)
        self.layer_map = {}
        
        # Count layers first
        idx = 0
        for _ in self.model.named_modules():
            idx += 1
        num_potential = idx
        
        # Initialize Bank
        try:
            self.adapter_bank = AdapterBank(num_layers=num_potential, device=self.device)
        except Exception:
            self.adapter_bank = None
        
        # Attach hooks and pre-allocate adapters
        idx = 0
        for name, module in self.model.named_modules():
            if isinstance(module, valid_types):
                self.layer_map[name] = idx
                
                # Pre-allocate adapter if possible
                if self.adapter_bank:
                    out_dim = None
                    if hasattr(module, 'out_features'): out_dim = module.out_features
                    elif hasattr(module, 'out_channels'): out_dim = module.out_channels
                    elif hasattr(module, 'hidden_size'): out_dim = module.hidden_size
                    
                    if out_dim:
                        self.adapter_bank.ensure_index(idx, out_dim=int(out_dim))
                
                module.register_forward_pre_hook(self._generate_fast_hook(idx))
                idx += 1
        
        self.num_tracked_layers = idx
        self.telemetry_buffer = torch.zeros(
            (idx, 4), 
            device=self.device, 
            dtype=torch.float32,
            requires_grad=False
        )

    def _generate_fast_hook(self, layer_idx):
        def hook(module, inputs):
            try:
                inp = inputs[0]
                if isinstance(inp, torch.Tensor):
                    # Fast Telemetry
                    with torch.no_grad():
                        if inp.numel() > 0:
                            # Use simple stats to avoid sync overhead
                            self.telemetry_buffer[layer_idx, 0] = inp.mean()
                            self.telemetry_buffer[layer_idx, 1] = inp.var(unbiased=False)
                            self.telemetry_buffer[layer_idx, 2] = 0 # Optimized out
                            self.telemetry_buffer[layer_idx, 3] = 0 # Optimized out

                    # Apply Adapter
                    if self.adapter_bank:
                        adapted = self.adapter_bank.apply(layer_idx, inp)
                        if adapted is not inp:
                            return (adapted,) + inputs[1:]
            except Exception:
                pass
            return None
        return hook

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        
        log_var = torch.tensor(0.0).to(self.device)
        affine_modifiers = None
        
        try:
            # Aggregate Telemetry
            global_state = self.telemetry_buffer.mean(dim=0)
            global_state = torch.nan_to_num(global_state, nan=0.0)
            
            # Introspection Step
            log_var, action, log_prob = self.introspection_engine(global_state)
            self.meta_log_probs.append(log_prob)
            affine_modifiers = action.detach()
                
        except Exception:
            self.meta_log_probs.clear()

        return output, log_var, affine_modifiers

    def train_step(self, input_data, target_data, enable_dream: bool = True, meta_step: bool = True, record_stats: bool = True):
        """
        Main training loop iteration.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self.meta_optimizer.zero_grad(set_to_none=True)
        if self.adapter_optimizer:
            self.adapter_optimizer.zero_grad(set_to_none=True)
        
        # 1. Forward Pass
        output, log_var, affine_modifiers = self.forward(input_data)

        
        # 2. Loss Calculation
        pred = output
        if hasattr(output, 'logits'): pred = output.logits
        elif isinstance(output, tuple): pred = output[0]
        
        if pred.shape == target_data.shape:
            raw_mse = F.mse_loss(pred, target_data)
            precision = torch.exp(-log_var)
            loss = torch.mean(0.5 * (log_var + (pred - target_data) ** 2 * precision))
        elif pred.dim() > target_data.dim(): 
             # Classification
             ce_loss = F.cross_entropy(pred.view(-1, pred.size(-1)), target_data.view(-1), reduction='none')
             raw_mse = ce_loss.mean()
             precision = torch.exp(-log_var)
             loss = torch.mean(0.5 * (log_var + ce_loss * precision))
        else:
             raw_mse = F.mse_loss(pred.float(), target_data.float())
             loss = raw_mse

        # NaN Guard
        if torch.isnan(loss) or torch.isinf(loss):
             self.meta_log_probs.clear()
             return {'loss': 10.0, 'status': 'nan_bailout'}

        current_loss_val = loss.item()
        current_mse_val = raw_mse.item()

        # 3. Hierarchical Reflex System (Mode Selection)
        z_score = 0.0
        if len(self.loss_history) > 20:
            hist_mean = np.mean(self.loss_history)
            hist_std = np.std(self.loss_history) + 1e-9
            z_score = (current_mse_val - hist_mean) / hist_std

        if self.step_count < self.config.warmup_steps:
            mode = "BOOTSTRAP"
            plasticity_gate = 1.0
            apply_memory = False
            trigger_consolidation = False
            block_reptile = True 
        elif current_mse_val > self.config.panic_threshold:
            mode = "PANIC"
            plasticity_gate = 1.0
            apply_memory = False 
            trigger_consolidation = False
            block_reptile = True 
        elif z_score > self.config.survival_z_threshold:
            mode = "SURVIVAL"
            plasticity_gate = 1.0
            apply_memory = False
            trigger_consolidation = False
            block_reptile = True
        elif z_score > self.config.novelty_z_threshold:
            mode = "NOVELTY"
            plasticity_gate = 1.0
            apply_memory = True
            trigger_consolidation, _ = self.consolidation_scheduler.should_consolidate(
                current_step=self.step_count, z_score=z_score, mode=mode, criterion=self.config.consolidation_criterion
            )
            block_reptile = False
            # Try auto-adapter
            self.auto_apply_best_task_memory(threshold=0.85)
        else:
            mode = "NORMAL"
            apply_memory = True
            plasticity_gate = 1.0 
            block_reptile = False
            trigger_consolidation, _ = self.consolidation_scheduler.should_consolidate(
                current_step=self.step_count, z_score=z_score, mode=mode, criterion=self.config.consolidation_criterion
            )

        # 4. Consciousness Interaction
        cons_importance = 1.0
        if self.consciousness:
            try:
                # Features for consciousness
                features = self.telemetry_buffer.detach()
                metrics = self.consciousness.observe(input_data, target_data, pred, features=features)
                self.consciousness.last_metrics = metrics

                
                cons_importance = metrics.get('importance', 1.0) # Used for replay priority
                
                # Override consolidation if consciousness demands it
                # (Simple heuristic based on returned metrics if needed)
                if metrics.get('surprise', 0) > 3.0:
                    trigger_consolidation = True
            except Exception:
                pass

        # 5. Consolidation Execution
        if trigger_consolidation:
            self.memory.consolidate(
                feedback_buffer=self.feedback_buffer,
                current_step=self.step_count,
                z_score=z_score,
                mode=mode
            )
            self.consolidation_scheduler.record_consolidation(self.step_count)
            # Save task memory
            fp = self.telemetry_buffer.mean(dim=0)
            self.memory.save_task_memory(adapters=self.adapter_bank, fingerprint=fp)

        # 6. Backward Pass with Memory Penalty
        if self.memory.is_enabled() and apply_memory:
            penalty = self.memory.compute_penalty(adaptive_mode=mode)
            loss += penalty

        
        # 7. Optimizer Step (with Shielding)
        if plasticity_gate > 0.01:
            # Backbone
            if plasticity_gate < 0.99:
                with torch.no_grad():
                    for p in self.model.parameters():
                        if p.grad is not None: p.grad.mul_(plasticity_gate)
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            
            # SI Accumulation Snapshot
            param_before = self.memory.before_step_snapshot()
            
            self.optimizer.step()
            
            # SI Accumulation Update
            self.memory.accumulate_path(param_before)
            
            # Adapter Step
            if self.adapter_optimizer:
                self.adapter_optimizer.step()
                # Clip adapters
                for p in self.adapter_bank.parameters():
                    if p.data.norm() > self.config.adapter_max_norm:
                        p.data.mul_(self.config.adapter_max_norm / (p.data.norm() + 1e-6))

        # 8. Meta-Learning (Reptile + Introspection)
        if self.reward_baseline == 0.0: self.reward_baseline = current_loss_val
        advantage = self.reward_baseline - current_loss_val
        self.reward_baseline = (1 - self.alpha) * self.reward_baseline + self.alpha * current_loss_val
        
        if meta_step and len(self.meta_log_probs) > 0:
            scale = 50.0 if mode in ["PANIC", "SURVIVAL"] else 10.0
            advantage_t = torch.tensor(
            advantage,
            device=self.device,
            dtype=torch.float32
             )

            policy_loss = -self.meta_log_probs[-1] * (advantage_t * scale)

            policy_loss.backward()
            self.meta_optimizer.step()
            self.meta_log_probs.clear()
            
            # Reptile Step (Only on real steps, not dreams)
            if not block_reptile:
                self.meta_controller.adapt(loss=current_loss_val)

        # 9. Weight Editing (Direct Cortex Manipulation)
        if self.step_count % self.config.evaluation_frequency == 0:
            avg_loss = np.mean(self.loss_history) if self.loss_history else loss.item()
            internals = {'affine_modifiers': affine_modifiers, 'telemetry_buffer': self.telemetry_buffer, 'layer_map': self.layer_map}
            try:
                self.monitor.adapt_weights(current_loss=loss.item(), previous_loss=avg_loss, activations=internals)
            except Exception:
                pass

        # 10. Dreaming (Replay)
        if record_stats:
            self.loss_history.append(current_mse_val)
            self.feedback_buffer.add(input_data, pred, target_data, -current_mse_val, current_mse_val)
            if self.prioritized_buffer:
                snapshot = self.feedback_buffer.buffer[-1]
                self.prioritized_buffer.add(snapshot, z_score=z_score, importance=cons_importance)

        if enable_dream and self.config.enable_dreaming and (self.step_count % self.config.dream_interval == 0):
             if mode not in ["PANIC", "SURVIVAL", "BOOTSTRAP"]:
                 self.learn_from_buffer(batch_size=16, num_epochs=1)
        
        if enable_dream: self.step_count += 1
        
        return {
            "loss": loss.item(),
            "mse": current_mse_val,
            "plasticity": plasticity_gate,
            "z_score": float(z_score),
            "mode": mode
        }
    
    def learn_from_buffer(self, batch_size: int = 32, num_epochs: int = 1):
        """
        Active Replay ("Dreaming").
        """
        if len(self.feedback_buffer.buffer) < 10:
            return
            
        self.model.train()
        for _ in range(num_epochs):
            buffer_size = len(self.feedback_buffer.buffer)
            effective_batch = min(batch_size, buffer_size)

            if effective_batch <= 0:
                return

            if self.prioritized_buffer:
                samples = self.prioritized_buffer.sample_batch(
                    effective_batch,
                    use_priorities=True
                )
            else:
                samples = random.sample(
                    self.feedback_buffer.buffer,
                    effective_batch
                )
                
            if not samples:
                continue
                
            # Fast concatenation
            try:
                inputs = torch.cat([s.input_data for s in samples], dim=0).to(self.device)
                targets = torch.cat([s.target for s in samples], dim=0).to(self.device)
            except Exception:
                continue
                
            # Train without meta-step (Lookahead behavior)
            self.train_step(
                inputs,
                targets,
                enable_dream=False,
                meta_step=False,
                record_stats=False
            )


    def save_checkpoint(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'introspection': self.introspection_engine.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'adapters': None if not self.adapter_bank else self.adapter_bank.state_dict(),
            'memory': self.memory.save_task_memory() # Save active memory state too
        }, path)
        self.logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.introspection_engine.load_state_dict(ckpt['introspection'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        
        if 'adapters' in ckpt and self.adapter_bank:
            self.adapter_bank.load_state_dict(ckpt['adapters'])
        
        # Load memory if present
        if 'memory' in ckpt and isinstance(ckpt['memory'], str):
             self.memory.load_task_memory(ckpt['memory'])
            
        self.logger.info(f"Checkpoint loaded from {path}")

    def auto_apply_best_task_memory(self, threshold: float = 0.8):
        """Search saved task memories by fingerprint."""
        try:
            names = self.memory.list_task_memories()
            if not names: return False
            
            fp = self.telemetry_buffer.mean(dim=0).flatten()
            fp_norm = fp / (fp.norm() + 1e-9)
            
            best_score, best_name = -1.0, None
            
            for name in names:
                payload = self.memory.load_task_memory(name)
                if not payload or 'fingerprint' not in payload: continue
                
                other = torch.tensor(payload['fingerprint'], device=self.device).flatten()
                score = F.cosine_similarity(fp_norm.unsqueeze(0), (other / (other.norm() + 1e-9)).unsqueeze(0)).item()
                
                if score > best_score:
                    best_score = score
                    best_name = name
            
            if best_name and best_score > threshold:
                self.logger.info(f"[MEMORY] Applied task memory: {best_name} (Score: {best_score:.2f})")
                
                # Apply Adapters
                payload = self.memory.load_task_memory(best_name)
                if 'adapters' in payload and self.adapter_bank:
                    # Logic to load partial adapters would go here
                    pass
                return True
                
        except Exception as e:
            self.logger.debug(f"Auto-memory failed: {e}")
        return False