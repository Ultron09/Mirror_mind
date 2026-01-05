"""
Core Adaptive Meta-Learning Framework (Universal V7.0 - "Production" Edition)
=============================================================================
The Universal Wrapper that turns ANY PyTorch model into a Self-Learning System.
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
from .moe import SparseMoE

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

    # --- V7.1: CORTEX ENGINE (MoE) ---
    use_moe: bool = False
    num_experts: int = 4
    top_k_experts: int = 2
    input_dim: int = 0 # Required for MoE gating if > 0. Else uses model_dim.

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
    input_args: tuple
    input_kwargs: dict
    output: torch.Tensor
    target: torch.Tensor
    reward: float
    loss: float
    timestamp: float
    episode: int
    
    def to_device(self, device):
        self.input_args = tuple(arg.to(device) for arg in self.input_args if isinstance(arg, torch.Tensor))
        self.input_kwargs = {k: v.to(device) for k, v in self.input_kwargs.items() if isinstance(v, torch.Tensor)}
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
        
    def add(self, input_args: tuple, input_kwargs: dict, output: torch.Tensor, target: torch.Tensor, reward: float, loss: float):
        # Move to CPU immediately to save VRAM
        snapshot = PerformanceSnapshot(
            input_args=tuple(arg.detach().cpu() for arg in input_args if isinstance(arg, torch.Tensor)),
            input_kwargs={k: v.detach().cpu() for k, v in input_kwargs.items() if isinstance(v, torch.Tensor)},
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
        log_var = torch.tanh(self.state_monitor(global_state))
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

        return abs(raw_scale) + abs(raw_shift)


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
        
        # 1. The "Body" (Base Model)
        self.model = user_model.to(self.device)
        
        # [V7.1] MoE Transformation
        if getattr(config, 'use_moe', False):
            moe_input_dim = config.input_dim if config.input_dim > 0 else config.model_dim
            self.logger.info(f"🧠 Transforming Cortex into Sparse MoE ({config.num_experts} Experts, Top-{config.top_k_experts})...")
            self.model = SparseMoE(
                base_model=self.model,
                input_dim=moe_input_dim,
                num_experts=config.num_experts,
                top_k=config.top_k_experts
            ).to(self.device)
            self.logger.info("   ✅ Transformation Complete. The Mind is now distributed.")

        # 2. Initialize Adapters & Hooks (CRITICAL: Must be before optimizer)
        self._init_adapters_and_hooks()
        
        # 3. The "Mind" (RL Policy)
        self.introspection_engine = IntrospectionEngine(
            input_dim=config.telemetry_dim
        ).to(self.device)
        
        # 4. The "Cortex" (Weight Editor)
        self.monitor = PerformanceMonitor(self.model, config, self.device)
        
        # 5. Memory System (Unified Handler V7.0)
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
        
        # 6. Experience Replay
        self.feedback_buffer = FeedbackBuffer(config, self.device)
        if getattr(config, 'use_prioritized_replay', True):
            self.prioritized_buffer = PrioritizedReplayBuffer(
                capacity=config.feedback_buffer_size,
                temperature=getattr(config, 'replay_priority_temperature', 0.6)
            )
        else:
            self.prioritized_buffer = None
        
        # 7. Adaptive Regularization & Consolidation
        self.adaptive_reg = AdaptiveRegularization(base_lambda=0.4)
        self.consolidation_scheduler = DynamicConsolidationScheduler(
            min_interval=getattr(config, 'consolidation_min_interval', 30),
            max_interval=getattr(config, 'consolidation_max_interval', 100)
        )
        
        # 8. Consciousness Layer
        if getattr(config, 'enable_consciousness', False):
            self.consciousness = ConsciousnessCore(
                feature_dim=config.model_dim,
                awareness_buffer_size=getattr(config, 'consciousness_buffer_size', 5000),
                novelty_threshold=getattr(config, 'novelty_threshold', 2.0)
            )
            self.logger.info("[CONSCIOUSNESS] Self-Awareness Module Active")
        else:
            self.consciousness = None
        
        # 9. Optimizers
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
                
                module.register_forward_hook(self._generate_fast_hook(idx, type(module)))
                idx += 1
        
        self.num_tracked_layers = idx
        self.telemetry_buffer = torch.zeros(
            (idx, 4), 
            device=self.device, 
            dtype=torch.float32,
            requires_grad=False
        )

    def _generate_fast_hook(self, layer_idx, module_type):
        def hook(module, inputs, output):
            try:
                inp = output
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
                        adapted = self.adapter_bank.apply(layer_idx, inp, module_type)
                        if adapted is not inp:
                            return adapted
            except Exception:
                pass
            return None
        return hook

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        
        # [V7.1] MoE Handling
        # If model is MoE, it returns (output, indices)
        moe_indices = None
        if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], torch.Tensor):
             # Check if second element looks like indices [B, k]
             if output[1].dtype == torch.long:
                 output, moe_indices = output
        
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

    def get_emotional_parameters(self, emotion: str) -> Tuple[float, bool, float]:
        """Map emotional state to learning parameters."""
        # Maps emotion to: (plasticity_gate, apply_memory, learning_rate_multiplier)
        params = {
            "confident": (1.0, True, 1.0),
            "anxious": (0.9, True, 1.2),
            "curious": (1.0, True, 1.1),
            "bored": (0.7, True, 0.8),
            "frustrated": (1.1, True, 1.5), # Keep memory penalty
            "satisfied": (1.0, True, 1.0),
            "overwhelmed": (0.5, True, 0.6), # Keep memory penalty
        }
        return params.get(emotion, (1.0, True, 1.0))

    def train_step(self, *model_inputs, target_data, enable_dream: bool = True, meta_step: bool = True, record_stats: bool = True):
        """
        Main training loop iteration for multi-input models.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self.meta_optimizer.zero_grad(set_to_none=True)
        if self.adapter_optimizer:
            self.adapter_optimizer.zero_grad(set_to_none=True)
        
        # 1. Forward Pass
        output, log_var, affine_modifiers = self.forward(*model_inputs)
        
        # 2. Loss Calculation
        pred = output
        latent_features = None
        if hasattr(output, 'logits'): 
            pred = output.logits
        elif isinstance(output, tuple): 
            pred = output[0]
            if len(output) > 1:
                latent_features = output[1]
        
        if pred.dim() > target_data.dim() and target_data.dim() == 1:
             loss = F.cross_entropy(pred.view(-1, pred.size(-1)), target_data.view(-1))
        else:
             loss = F.mse_loss(pred.float(), target_data.float())

        if torch.isnan(loss) or torch.isinf(loss):
             self.meta_log_probs.clear()
             return {'loss': 10.0, 'status': 'nan_bailout'}

        current_loss_val = loss.item()
        current_mse_val = current_loss_val

        # 3. Consciousness Interaction
        cons_metrics = {}
        if self.consciousness:
            try:
                # Use latent features if available, otherwise fallback to input or telemetry
                if latent_features is not None:
                    features_for_cons = latent_features
                elif len(model_inputs) > 0 and isinstance(model_inputs[0], torch.Tensor):
                    features_for_cons = model_inputs[0]
                else:
                    features_for_cons = self.telemetry_buffer.detach()
                
                cons_metrics = self.consciousness.observe(*model_inputs, y_true=target_data, y_pred=pred, features=features_for_cons)
                self.consciousness.last_metrics = cons_metrics
            except Exception as e:
                self.logger.debug(f"Consciousness observe failed: {e}")
                cons_metrics['emotion'] = 'confident' # Default
        else:
            cons_metrics['emotion'] = 'confident'
            cons_metrics['surprise'] = 0.0

        # 4. Emotional Control
        emotion = cons_metrics.get('emotion', 'confident')
        # Use dynamic multiplier if available (V7.2+), otherwise fallback to static map
        if 'learning_rate_multiplier' in cons_metrics:
            lr_multiplier = cons_metrics['learning_rate_multiplier']
            plasticity_gate, apply_memory, _ = self.get_emotional_parameters(emotion)
        else:
            plasticity_gate, apply_memory, lr_multiplier = self.get_emotional_parameters(emotion)
        
        # [NEW] REFLEX TRIGGER (V7.3)
        current_surprise = cons_metrics.get('surprise', 0.0)
        if current_surprise > 3.0:
            if self.step_count % 10 == 0: 
                 self.logger.warning(f"⚡ REFLEX TRIGGERED: Surprise Z-Score {current_surprise:.2f} > 3.0. Boosting Plasticity (10x)!")
            lr_multiplier = 10.0 
            plasticity_gate = True

        # Adjust LR
        for g in self.optimizer.param_groups:
            g['lr'] = self.config.learning_rate * lr_multiplier

        # 5. Backward Pass & Optimization
        if plasticity_gate:
            loss.backward(retain_graph=True) # Retain for meta-step
            self.optimizer.step()
            
            if self.adapter_optimizer:
                self.adapter_optimizer.step()

        # 8. Meta-Learning
        if self.reward_baseline == 0.0: self.reward_baseline = current_loss_val
        advantage = self.reward_baseline - current_loss_val
        self.reward_baseline = (1 - self.alpha) * self.reward_baseline + self.alpha * current_loss_val
        
        if meta_step and len(self.meta_log_probs) > 0:
            advantage_t = torch.tensor(advantage, device=self.device, dtype=torch.float32)
            policy_loss = -self.meta_log_probs[-1] * advantage_t
            policy_loss.backward()
            self.meta_optimizer.step()
            self.meta_log_probs.clear()
            
            if emotion not in ['frustrated', 'overwhelmed']:
                self.meta_controller.adapt(loss=current_loss_val)

        # 9. Weight Editing
        if self.step_count % self.config.evaluation_frequency == 0:
            avg_loss = np.mean(self.loss_history) if self.loss_history else loss.item()
            internals = {'affine_modifiers': affine_modifiers, 'telemetry_buffer': self.telemetry_buffer, 'layer_map': self.layer_map}
            self.monitor.adapt_weights(current_loss=loss.item(), previous_loss=avg_loss, activations=internals)

        # 10. Dreaming & Stats
        if record_stats:
            self.loss_history.append(current_mse_val)
            self.feedback_buffer.add(model_inputs, {}, pred, target_data, -current_mse_val, current_mse_val)
            if self.prioritized_buffer:
                snapshot = self.feedback_buffer.buffer[-1]
                self.prioritized_buffer.add(snapshot, z_score=cons_metrics.get('surprise', 0), importance=cons_metrics.get('importance', 1.0))

        if enable_dream and self.config.enable_dreaming and (self.step_count % self.config.dream_interval == 0):
             if emotion not in ['overwhelmed', 'frustrated']:
                 self.learn_from_buffer(batch_size=16, num_epochs=1)
        
        # [NEW] Episodic Replay Trigger (V7.2)
        if enable_dream and self.consciousness and (self.step_count % self.config.dream_interval == 0):
             # Use current surprise/loss to query relevant memories
             # [OPTIMIZATION] Pass features for content-aware retrieval
             # Use latent_features if available, otherwise None (telemetry is bad for content retrieval)
             features_for_replay = latent_features if latent_features is not None else None
             
             self.learn_from_episodic_memory(
                 current_surprise=cons_metrics.get('surprise', 0.0),
                 current_loss=cons_metrics.get('loss', 0.0),
                 current_features=features_for_replay
             )
        
        if enable_dream: self.step_count += 1
        
        return {
            "loss": loss.item(),
            "mse": current_mse_val,
            "plasticity": plasticity_gate,
            "z_score": float(cons_metrics.get('surprise', 0)),
            "mode": emotion
        }

    def learn_from_buffer(self, batch_size: int = 32, num_epochs: int = 1):
        """
        Active Replay ("Dreaming") for multi-input models.
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
                
            # --- New Batching Logic for Multi-Input Models ---
            try:
                # Transpose the list of input_args tuples
                # Assumes all experiences in the buffer have the same number of input args
                num_args = len(samples[0].input_args)
                batch_args = []
                for i in range(num_args):
                    # For each argument position, concatenate the tensors from all samples
                    arg_tensors = [s.input_args[i].to(self.device) for s in samples]
                    batch_args.append(torch.cat(arg_tensors, dim=0))
                
                batch_targets = torch.cat([s.target.to(self.device) for s in samples], dim=0)

            except Exception as e:
                self.logger.debug(f"Failed to create replay batch, skipping dream step: {e}")
                continue
                
            # Call train_step with unpacked arguments
            self.train_step(
                *batch_args,
                target_data=batch_targets,
                enable_dream=False,
                meta_step=False,
                record_stats=False
            )

    def learn_from_episodic_memory(self, current_surprise: float, current_loss: float, current_features: Optional[torch.Tensor] = None, k: int = 5):
        """
        Replay specific, relevant episodes from consciousness.
        """
        if not self.consciousness: return

        # 1. Retrieve
        memories = self.consciousness.episodic_memory.retrieve_relevant_memories(
            current_surprise=current_surprise,
            current_error=current_loss,
            current_features=current_features,
            k=k
        )
        
        if not memories: return

        # 2. Construct Batch
        try:
            valid_memories = [m for m in memories if m.y is not None and m.x is not None]
            if not valid_memories: return

            # Stack inputs and targets
            # NOTE: Currently supports single-input models for episodic replay
            batch_x = torch.stack([m.x.to(self.device) for m in valid_memories])
            batch_y = torch.stack([m.y.to(self.device) for m in valid_memories])
            
            # 3. Replay
            self.train_step(
                batch_x, 
                target_data=batch_y,
                enable_dream=False,
                meta_step=False,
                record_stats=False
            )
            
        except Exception as e:
            self.logger.debug(f"Episodic replay failed: {e}")

    def save_checkpoint(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'config': self.config,  # Save the configuration
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