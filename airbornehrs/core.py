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
import os
import platform
import shutil
from datetime import datetime
import time

# Import EWC, Memory, Meta-Controller, and Consciousness

from .ewc import EWCHandler, SIHandler
from .memory import UnifiedMemoryHandler, PrioritizedReplayBuffer, AdaptiveRegularization, DynamicConsolidationScheduler
from .meta_controller import MetaController, MetaControllerConfig
from .consciousness import ConsciousnessCore, AttentionMechanism, IntrinisicMotivation, SelfAwarenessMonitor

# OPTIMIZATION: Use Tensor Cores on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

# ==================== CONFIGURATION ====================

@dataclass
class AdaptiveFrameworkConfig:
    """
    Configuration for the Universal Framework (V6.5 - Panic Switch Edition).
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
    # How often to run dreaming/replay (in steps). Increase for short-run stability.
    dream_interval: int = 10
    
    # Optimization
    compile_model: bool = True 
    use_amp: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_frequency: int = 50
    checkpoint_frequency: int = 500
    # Gradient clipping
    gradient_clip_norm: float = 1.0
    # Maximum allowed norm for adapter parameters (prevents catastrophic adapter jumps)
    adapter_max_norm: float = 2.0
    
    # --- V6.5: HIERARCHICAL REFLEX ---
    enable_active_shield: bool = True 
    
    # SHIELD: Only activate shield if error is BELOW this.
    active_shield_threshold: float = 0.05 
    # SLOPE: Lowered from 50.0 to 10.0 to prevent "Snap-Freezing" adaptation.
    active_shield_slope: float = 10.0   
    
    # PANIC: If Raw MSE > 0.2, IGNORE Z-SCORES. JUST ADAPT.
    # This is the "Gravity Failed" override.
    panic_threshold: float = 0.2
    
    # Bootstrapping: Force learning for first N steps (Fixes the Step 20 Crash)
    warmup_steps: int = 50
    
    # Z-Score Thresholds (Statistical Surprise)
    novelty_z_threshold: float = 2.0
    survival_z_threshold: float = 4.0
    # Allow experiments to disable dreaming/replay to improve short-run stability
    enable_dreaming: bool = True
    # Tracing / debugging
    enable_tracing: bool = False
    trace_max_records: int = 1000
    # Importance estimation method: 'ewc' (default) or 'si' (synaptic intelligence)
    importance_method: str = 'ewc'
    # SI hyperparameters (used if importance_method == 'si')
    si_lambda: float = 1.0
    si_xi: float = 1e-3
    
    # SOTA Unified Memory System (V7.0)
    memory_type: str = 'hybrid'  # 'ewc', 'si', or 'hybrid'
    consolidation_criterion: str = 'hybrid'  # 'time', 'surprise', or 'hybrid'
    consolidation_min_interval: int = 30  # Min steps before consolidation allowed
    consolidation_max_interval: int = 100  # Max steps between consolidations
    consolidation_surprise_threshold: float = 2.5  # Z-score threshold for surprise-based consolidation
    adaptive_lambda: bool = True  # Scale λ by operating mode
    use_prioritized_replay: bool = True  # Prioritize hard/surprising examples
    replay_priority_temperature: float = 0.6  # Softmax temperature for priority sampling (0=greedy, 1=uniform)
    
    # --- V7.0: CONSCIOUSNESS LAYER ---
    enable_consciousness: bool = True  # Enable self-aware learning
    use_attention: bool = True  # Learn which features matter
    use_intrinsic_motivation: bool = True  # Learn from curiosity/uncertainty
    consciousness_buffer_size: int = 5000  # How much history to track
    novelty_threshold: float = 2.0  # Z-score threshold for "novel" examples

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
            enable_consciousness=True,
            use_attention=True,
            use_intrinsic_motivation=True
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
        # Guard against NaNs/Infs in policy outputs
        policy_out = torch.nan_to_num(policy_out, nan=0.0, posinf=10.0, neginf=-10.0)

        # Split into Mu and Log-Sigma (using log for numerical stability)
        # Shape: [Batch, 4] -> [Batch, 2], [Batch, 2]
        try:
            mu, log_sigma = policy_out.chunk(2, dim=-1)
        except Exception:
            # Fallback to zeros
            mu = torch.zeros(1, 2, device=global_state.device)
            log_sigma = torch.zeros(1, 2, device=global_state.device)

        # Clamp log_sigma and convert to sigma with safety floor
        log_sigma = torch.clamp(log_sigma, min=-10.0, max=5.0)
        sigma = torch.exp(log_sigma)
        sigma = torch.clamp(sigma, min=1e-3, max=10.0)

        # Create Distribution (guarded)
        try:
            dist = torch.distributions.Normal(mu, sigma)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        except Exception:
            # Return safe defaults on failure
            action = torch.zeros_like(mu)
            log_prob = torch.zeros(mu.size(0), device=mu.device)

        # Expose last policy params for tracing (non-blocking, small objects)
        try:
            # store cpu numpy copies for later inspection
            self._last_mu = mu.detach().cpu().numpy() if isinstance(mu, torch.Tensor) else None
            self._last_sigma = sigma.detach().cpu().numpy() if isinstance(sigma, torch.Tensor) else None
        except Exception:
            self._last_mu = None
            self._last_sigma = None

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
        
        # 3. Memory System (SOTA V7.0 - Unified Handler)
        memory_type = getattr(config, 'memory_type', 'hybrid')
        consolidation_criterion = getattr(config, 'consolidation_criterion', 'hybrid')
        
        try:
            if memory_type in ['si', 'hybrid']:
                # Use new unified handler with SI
                self.ewc = UnifiedMemoryHandler(
                    self.model,
                    method=memory_type,
                    si_lambda=getattr(config, 'si_lambda', 1.0),
                    si_xi=getattr(config, 'si_xi', 1e-3),
                    ewc_lambda=0.4,
                    consolidation_criterion=consolidation_criterion
                )
                self.logger.info(f"[BRAIN] Using Unified Memory Handler (method={memory_type}, consolidation={consolidation_criterion})")
            else:
                # Fall back to legacy EWC
                self.ewc = EWCHandler(self.model, ewc_lambda=0.4)
                self.logger.info("🔁 Using legacy EWC handler")
        except Exception as e:
            self.logger.warning(f"Memory handler initialization failed, fallback to EWC: {e}")
            self.ewc = EWCHandler(self.model, ewc_lambda=0.4)
        
        # 4. Experience Replay (with optional prioritization)
        self.feedback_buffer = FeedbackBuffer(config, self.device)
        use_prioritized = getattr(config, 'use_prioritized_replay', True)
        if use_prioritized:
            self.prioritized_buffer = PrioritizedReplayBuffer(
                capacity=config.feedback_buffer_size,
                temperature=getattr(config, 'replay_priority_temperature', 0.6)
            )
            self.logger.info("[REPLAY] Prioritized replay enabled")
        else:
            self.prioritized_buffer = None
        
        # 5. Adaptive Regularization
        self.adaptive_reg = AdaptiveRegularization(base_lambda=0.4)
        
        # 6. Consolidation Scheduler
        self.consolidation_scheduler = DynamicConsolidationScheduler(
            min_interval=getattr(config, 'consolidation_min_interval', 30),
            max_interval=getattr(config, 'consolidation_max_interval', 100)
        )
        
        # 7. CONSCIOUSNESS LAYER (V7.0 - Self-Aware Learning)
        # BUG FIX #12: Default to False for backward compatibility
        enable_consciousness = getattr(config, 'enable_consciousness', False)
        if enable_consciousness:
            self.consciousness = ConsciousnessCore(
                model=self.model,
                feature_dim=config.model_dim,
                awareness_buffer_size=getattr(config, 'consciousness_buffer_size', 5000),
                novelty_threshold=getattr(config, 'novelty_threshold', 2.0)
            )
            self.logger.info("[CONSCIOUSNESS] Consciousness layer enabled (self-aware learning)")
            
            # Attention mechanism (learns which features matter)
            if getattr(config, 'use_attention', True):
                self.attention = AttentionMechanism(
                    feature_dim=config.model_dim,
                    num_heads=config.num_heads,
                    learned=True
                ).to(self.device)
                self.logger.info("[ATTENTION] Attention mechanism enabled (feature importance learning)")
            else:
                self.attention = None
            
            # Intrinsic motivation (curiosity-driven learning)
            if getattr(config, 'use_intrinsic_motivation', True):
                self.intrinsic_motivation = IntrinisicMotivation(
                    update_frequency=100,
                    uncertainty_weight=0.5,
                    novelty_weight=0.3,
                    learning_progress_weight=0.2
                )
                self.logger.info("[MOTIVATION] Intrinsic motivation enabled (curiosity-driven learning)")
            else:
                self.intrinsic_motivation = None
            
            # Self-awareness monitor
            self.self_awareness = SelfAwarenessMonitor(moving_average_window=100)
            self.logger.info("[AWARENESS] Self-awareness monitor enabled")
        else:
            self.consciousness = None
            self.attention = None
            self.intrinsic_motivation = None
            self.self_awareness = None
        
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
        # Adapter optimizer: small, fast-learning adapters
        try:
            adapter_params = list(self.adapter_bank.parameters()) if hasattr(self, 'adapter_bank') and self.adapter_bank is not None else []
            if adapter_params:
                self.adapter_optimizer = AdamW(adapter_params, lr=config.weight_adaptation_lr)
            else:
                self.adapter_optimizer = None
        except Exception:
            self.adapter_optimizer = None
        
        self.loss_history = deque(maxlen=100) # Increased for Z-Score calc
        self.meta_log_probs = []
        self.step_count = 0
        # Step-level tracing (populated when MM_TRACE=1)
        self.step_trace = []
        
        # RL Initialization
        self.reward_baseline = 0.0
        self.alpha = 0.1
        
        # Compilation
        if config.compile_model and hasattr(torch, 'compile'):
            is_windows = platform.system() == 'Windows'
            has_cl = shutil.which('cl') is not None
            
            if is_windows and not has_cl:
                self.logger.warning("[WARNING] Windows detected without C++ Compiler. Disabling torch.compile.")
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
                # Use forward pre-hook to apply adapters to inputs safely (avoids inplace on outputs)
                module.register_forward_pre_hook(self._generate_fast_hook(idx))
                idx += 1
        
        self.num_tracked_layers = idx
        self.telemetry_buffer = torch.zeros(
            (self.num_tracked_layers, 4), 
            device=self.device, 
            dtype=torch.float32,
            requires_grad=False
        )
        # Initialize AdapterBank for fast, parameter-efficient task adapters
        try:
            from .adapters import AdapterBank
            self.adapter_bank = AdapterBank(num_layers=self.num_tracked_layers, device=self.device)
            self.logger.info(f"[ADAPTER] AdapterBank initialized for {self.num_tracked_layers} layers.")
        except Exception as e:
            self.adapter_bank = None
            self.logger.warning(f"AdapterBank initialization failed: {e}")
        self.logger.info(f"[TELEMETRY] Fast Telemetry Bus established for {idx} layers.")

    def _generate_fast_hook(self, layer_idx):
        def hook(module, inputs):
            # inputs is a tuple; we will examine first tensor input and apply adapter to it
            try:
                inp = inputs[0]
                if isinstance(inp, torch.Tensor):
                    act_flat = inp.detach().flatten()
                    if act_flat.numel() > 0:
                        self.telemetry_buffer[layer_idx, 0] = act_flat.mean()
                        self.telemetry_buffer[layer_idx, 1] = act_flat.var(unbiased=False)
                        self.telemetry_buffer[layer_idx, 2] = act_flat.max()
                        self.telemetry_buffer[layer_idx, 3] = act_flat.min()

                    # Apply adapter out-of-place and return modified inputs tuple
                    if hasattr(self, 'adapter_bank') and self.adapter_bank is not None:
                        # Try to infer output dim from module attributes for vector adapters
                        out_dim = None
                        try:
                            if hasattr(module, 'out_features'):
                                out_dim = int(module.out_features)
                            elif hasattr(module, 'out_channels'):
                                out_dim = int(module.out_channels)
                            elif hasattr(module, 'hidden_size'):
                                out_dim = int(module.hidden_size)
                        except Exception:
                            out_dim = None

                        # Ensure adapter storage is sized appropriately
                        try:
                            self.adapter_bank.ensure_index(layer_idx, out_dim=out_dim)
                        except Exception:
                            pass

                        adapted = self.adapter_bank.apply(layer_idx, inp)
                        if adapted is not inp:
                            # return a new inputs tuple with adapted first element
                            new_inputs = (adapted,) + inputs[1:]
                            return new_inputs
            except Exception:
                pass
            # default: no change
            return None
        return hook

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        
        log_var = torch.tensor(0.0).to(self.device)
        affine_modifiers = None
        
        try:
            # FAST PATH: No stacking needed. The buffer IS the state.
            global_state = self.telemetry_buffer.mean(dim=0)
            # Guard against NaNs/Infs in telemetry (can happen early or with bad hooks)
            global_state = torch.nan_to_num(global_state, nan=0.0, posinf=1e3, neginf=-1e3)
            # Clamp to a reasonable dynamic range to stabilize Introspection input
            global_state = torch.clamp(global_state, min=-10.0, max=10.0)

            # RL Step (guarded)
            try:
                log_var, action, log_prob = self.introspection_engine(global_state)
                # Ensure outputs are finite
                if not torch.isfinite(log_var).all():
                    raise ValueError('Introspection produced non-finite log_var')
                if action is None or not torch.isfinite(action).all():
                    raise ValueError('Introspection produced invalid action')
                self.meta_log_probs.append(log_prob)
                affine_modifiers = action.detach()
            except Exception as e:
                self.logger.debug(f"Introspection failed (guarded): {e}")
                log_var = torch.tensor(0.0).to(self.device)
                affine_modifiers = None
                
        except Exception as e:
            self.logger.warning(f"Introspection failed: {e}")
            log_var = torch.tensor(0.0).to(self.device)
            affine_modifiers = None
            self.meta_log_probs.clear()
        # If tracing is enabled at inference, write a light-weight forward trace
        try:
            if os.environ.get('MM_TRACE', '0') == '1':
                try:
                    import json
                    dbg_dir = Path('debug')
                    dbg_dir.mkdir(parents=True, exist_ok=True)
                    seed = os.environ.get('MM_SEED', None) or str(int(time.time()))
                    fpath = dbg_dir / f'forward_trace_{seed}.ndjson'
                    entry = {
                        'step_count': int(getattr(self, 'step_count', 0)),
                        'log_var': float(log_var.item()) if hasattr(log_var, 'item') else None,
                        'affine_modifiers': None if affine_modifiers is None else (affine_modifiers.detach().cpu().tolist() if hasattr(affine_modifiers, 'detach') else None),
                        'telemetry': None
                    }
                    try:
                        entry['telemetry'] = self.telemetry_buffer.detach().cpu().tolist()
                    except Exception:
                        entry['telemetry'] = None
                    self.logger.debug(f"[TRACE] Writing forward trace to {fpath}")
                    with open(fpath, 'a', encoding='utf-8') as fh:
                        fh.write(json.dumps(entry) + '\n')
                except Exception:
                    pass
        except Exception:
            pass

        return output, log_var, affine_modifiers

    def train_step(self, input_data, target_data, enable_dream: bool = True, meta_step: bool = True):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self.meta_optimizer.zero_grad(set_to_none=True)
        if hasattr(self, 'adapter_optimizer') and self.adapter_optimizer is not None:
            try:
                self.adapter_optimizer.zero_grad(set_to_none=True)
            except Exception:
                pass
        
        # 1. Forward Pass
        output, log_var, affine_modifiers = self.forward(input_data)
        
        # Loss & Raw MSE Calculation
        pred = output
        if hasattr(output, 'logits'): pred = output.logits
        elif isinstance(output, tuple): pred = output[0]
        
        # Calculate Raw MSE (The Absolute Truth)
        if pred.shape == target_data.shape:
            raw_mse = F.mse_loss(pred, target_data)
            precision = torch.exp(-log_var)
            loss = torch.mean(0.5 * (log_var + (pred - target_data) ** 2 * precision))
        elif pred.dim() > target_data.dim(): 
             # Classification Support
             ce_loss = F.cross_entropy(pred.view(-1, pred.size(-1)), target_data.view(-1), reduction='none')
             raw_mse = ce_loss.mean()
             precision = torch.exp(-log_var)
             loss = torch.mean(0.5 * (log_var + ce_loss * precision))
        else:
             raw_mse = F.mse_loss(pred.float(), target_data.float())
             loss = raw_mse

        # NaN Guard
        if torch.isnan(loss) or torch.isinf(loss):
             if len(self.meta_log_probs) > 0: self.meta_log_probs.pop()
             return {'loss': 10.0, 'status': 'nan_bailout'}

        current_loss_val = loss.item()
        current_mse_val = raw_mse.item()

        # --- Tracing: collect pre-update param/adapters norms and telemetry if requested
        do_trace = os.environ.get('MM_TRACE', '0') == '1'
        trace_entry = None
        if do_trace:
            try:
                # model param norm
                total_param_norm = 0.0
                for p in self.model.parameters():
                    try:
                        total_param_norm += float(p.data.norm().item() ** 2)
                    except Exception:
                        pass
                total_param_norm = float(total_param_norm ** 0.5)

                # adapter norms (if any)
                adapter_norms = None
                if hasattr(self, 'adapter_bank') and self.adapter_bank is not None:
                    adapter_norms = []
                    try:
                        for p in self.adapter_bank.parameters():
                            try:
                                adapter_norms.append(float(p.data.norm().item()))
                            except Exception:
                                adapter_norms.append(None)
                    except Exception:
                        adapter_norms = None

                telemetry_snapshot = None
                try:
                    telemetry_snapshot = self.telemetry_buffer.detach().cpu().numpy().tolist()
                except Exception:
                    telemetry_snapshot = None

                trace_entry = {
                    'step': int(self.step_count),
                    'pre_param_norm': total_param_norm,
                    'pre_adapter_norms': adapter_norms,
                    'telemetry': telemetry_snapshot,
                    'affine_modifiers': None if affine_modifiers is None else (affine_modifiers.detach().cpu().numpy().tolist() if hasattr(affine_modifiers, 'detach') else None),
                    'log_var': float(log_var.item()) if hasattr(log_var, 'item') else None,
                    'loss_pre': float(current_loss_val),
                    'mse_pre': float(current_mse_val)
                }
            except Exception:
                trace_entry = None
        
        # ---------------------------------------------------------
        # V6.5: HIERARCHICAL REFLEX SYSTEM
        # ---------------------------------------------------------
        
        # A. Calculate Statistical Surprise (Z-Score)
        z_score = 0.0
        if len(self.loss_history) > 20:
            hist_mean = np.mean(self.loss_history)
            hist_std = np.std(self.loss_history) + 1e-9
            z_score = (current_mse_val - hist_mean) / hist_std

        # Initialize consciousness signals (defaults if consciousness is disabled)
        consciousness_urgency = 0.0
        cons_importance = 1.0

        # B. MODE SELECTION (The Hierarchy)
        
        # 1. BOOTSTRAP: Always learn at start (Warmup)
        # FIXES STEP 20 CRASH: Forces learning regardless of shield status.
        if self.step_count < self.config.warmup_steps:
            mode = "BOOTSTRAP"
            plasticity_gate = 1.0
            apply_ewc = False
            trigger_consolidation = False
            block_reptile = True 
            
        # 2. PANIC: Absolute Error is too high (Gravity Failure)
        # FIXES STEP 50 CRASH: Overrides safety checks if we are falling.
        elif current_mse_val > self.config.panic_threshold:
            mode = "PANIC"
            plasticity_gate = 1.0
            apply_ewc = False # CUT THE TETHER (Don't fight old memories)
            trigger_consolidation = False
            block_reptile = True # Stop Reptile from pulling weights back
            
        # 3. SURVIVAL: Error is low, but spiked massively (Sudden Shock)
        elif z_score > self.config.survival_z_threshold:
            mode = "SURVIVAL"
            plasticity_gate = 1.0
            apply_ewc = False
            trigger_consolidation = False
            block_reptile = True
            
        # 4. NOVELTY: New pattern detected (Bird on drone)
        elif z_score > self.config.novelty_z_threshold:
            mode = "NOVELTY"
            plasticity_gate = 1.0
            apply_ewc = True # Enable Memory Protection
            
            # Use dynamic consolidation scheduler (SOTA V7.0)
            trigger_consolidation, reason = self.consolidation_scheduler.should_consolidate(
                current_step=self.step_count,
                z_score=z_score,
                mode=mode,
                criterion=getattr(self.config, 'consolidation_criterion', 'hybrid')
            )
            self.logger.debug(f"Consolidation check (NOVELTY): {reason}")
            
            block_reptile = False
            # Attempt to retrieve a matching task memory and apply its adapter
            try:
                # Use a fairly high similarity threshold for auto-apply
                applied = self.auto_apply_best_task_memory(threshold=0.85)
                if applied:
                    self.logger.info("[ADAPTER] Adapter auto-applied from TaskMemory (novelty match).")
            except Exception as e:
                self.logger.debug(f"Auto-retrieval failed: {e}")
            
        # 5. NORMAL: Smooth sailing -> Engage Active Shield
        else:
            mode = "NORMAL"
            apply_ewc = True
            plasticity_gate = 1.0  # Initialize for NORMAL mode (will be modulated by Active Shield)
            block_reptile = False  # Initialize for NORMAL mode
            
            # Use dynamic consolidation scheduler for NORMAL mode too
            trigger_consolidation, reason = self.consolidation_scheduler.should_consolidate(
                current_step=self.step_count,
                z_score=z_score,
                mode=mode,
                criterion=getattr(self.config, 'consolidation_criterion', 'hybrid')
            )
            self.logger.debug(f"Consolidation check (NORMAL): {reason}")
        
        # --- CONSCIOUSNESS OBSERVATION (EARLY OVERRIDE) ---
        # Call consciousness BEFORE consolidation action so urgency can override scheduler
        try:
            if getattr(self.config, 'enable_consciousness', False) and getattr(self, 'consciousness', None) is not None:
                # Attempt to extract features for consciousness
                features = None
                try:
                    if hasattr(self, 'telemetry_buffer'):
                        features = self.telemetry_buffer.detach() if hasattr(self.telemetry_buffer, 'detach') else None
                except Exception:
                    features = None
                
                # Observe this example (updates internal gap/surprise stats)
                try:
                    cons_metrics = self.consciousness.observe(input_data, target_data, pred, features=features)
                except Exception as e:
                    self.logger.debug(f"Consciousness observe failed: {e}")
                    cons_metrics = {'confidence': 0.5, 'uncertainty': 0.5, 'surprise': 0.0, 'importance': 1.0}
                
                # Extract importance signal for replay prioritization
                cons_importance = cons_metrics.get('importance', 1.0)
                
                # Update self-awareness monitor if present
                try:
                    if getattr(self, 'self_awareness', None) is not None:
                        self.self_awareness.update(
                            confidence=cons_metrics.get('confidence', 0.5),
                            accuracy=1.0 - current_mse_val if current_mse_val < 1.0 else 0.0,
                            learning_gap=self.consciousness.get_knowledge_state().get('learning_gap', 0.5),
                            surprise=cons_metrics.get('surprise', 0.0)
                        )
                except Exception:
                    pass
                
                # Get learning priority to influence consolidation and replay
                try:
                    priority = self.consciousness.get_learning_priority()
                    consciousness_urgency = priority.get('consolidation_urgency', 0.0)
                    
                    # Boost prioritized replay sampling temperature or weights
                    if getattr(self, 'prioritized_buffer', None) is not None:
                        try:
                            # Adjust temperature toward exploitation when priority high
                            t = max(0.1, min(1.0, getattr(self.config, 'replay_priority_temperature', 0.6) * (1.0 - priority.get('replay_priority', 0.5))))
                            self.prioritized_buffer.temperature = t
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            # Consciousness is best-effort; do not break training
            pass
        
        # Override consolidation trigger if consciousness urgency is high
        if consciousness_urgency > 0.8:
            trigger_consolidation = True
            self.logger.info(f"[CONSCIOUSNESS] Consciousness Override: Consolidation urgency={consciousness_urgency:.2f}")
            
            block_reptile = False
            
            if self.config.enable_active_shield:
                # Calculate Shield Strength: If Error is low, clamp plasticity.
                delta = current_mse_val - self.config.active_shield_threshold
                plasticity_gate = torch.sigmoid(torch.tensor(delta * self.config.active_shield_slope)).item()
            else:
                plasticity_gate = 1.0

        # C. EXECUTION - Smart Consolidation with Unified Handler
        if trigger_consolidation:
            # Use unified consolidate API with new parameters
            try:
                if hasattr(self.ewc, 'consolidate'):
                    self.ewc.consolidate(
                        feedback_buffer=self.feedback_buffer,
                        current_step=self.step_count,
                        z_score=z_score,
                        mode=mode
                    )
                    self.consolidation_scheduler.record_consolidation(self.step_count)
                elif hasattr(self.ewc, 'consolidate_from_buffer'):
                    self.ewc.consolidate_from_buffer(self.feedback_buffer)
            except Exception as e:
                self.logger.warning(f"Consolidation failed: {e}")
            # Save adapters and fingerprint together with task memory for later retrieval
            try:
                fp = self.telemetry_buffer.mean(dim=0) if hasattr(self, 'telemetry_buffer') else None
                self.ewc.save_task_memory(name=None, data_loader=None, adapters=getattr(self, 'adapter_bank', None), fingerprint=fp)
            except Exception as e:
                self.logger.warning(f"Failed to save task memory with adapters: {e}")

            self.loss_history.clear()

        # Backward Pass with Adaptive Memory Penalty
        # Only apply Memory Penalty (EWC/SI) if NOT in Panic/Survival
        if self.ewc.is_enabled() and apply_ewc:
            # Compute adaptive lambda based on operating mode (SOTA V7.0)
            if hasattr(self, 'adaptive_reg') and getattr(self.config, 'adaptive_lambda', True):
                # Track steps in current mode
                if not hasattr(self, 'mode_history'):
                    self.mode_history = deque(maxlen=10)
                    self.mode_step_count = 0
                
                if len(self.mode_history) == 0 or self.mode_history[-1] != mode:
                    self.mode_history.append(mode)
                    self.mode_step_count = 0
                else:
                    self.mode_step_count += 1
                
                # Get adaptive lambda
                adaptive_lambda_mult = self.adaptive_reg.get_lambda(mode, self.mode_step_count)
            else:
                adaptive_lambda_mult = 1.0
            
            # Compute penalty with adaptive multiplier
            if hasattr(self.ewc, 'compute_penalty') and callable(getattr(self.ewc, 'compute_penalty')):
                penalty = self.ewc.compute_penalty(adaptive_mode=mode, step_in_mode=getattr(self, 'mode_step_count', 0))
            else:
                penalty = self.ewc.compute_penalty()
            
            loss += penalty * adaptive_lambda_mult

        loss.backward(retain_graph=True)

        # record gradient norm (pre-clipping)
        if do_trace and trace_entry is not None:
            try:
                grad_sq = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        try:
                            grad_sq += float(p.grad.data.norm().item() ** 2)
                        except Exception:
                            pass
                grad_norm = float(grad_sq ** 0.5)
                trace_entry['grad_norm_preclip'] = grad_norm
            except Exception:
                trace_entry['grad_norm_preclip'] = None

        # Adapter updates: always allow adapters to step (fast adaptation),
        # but in emergency modes we skip backbone updates (adapter-only).
        adapter_only = False
        if hasattr(self, 'adapter_optimizer') and self.adapter_optimizer is not None and mode in ["PANIC", "SURVIVAL"]:
            adapter_only = True

        # Prepare SI snapshot before any optimizer steps (if supported)
        param_before = None
        try:
            if hasattr(self.ewc, 'before_step_snapshot'):
                param_before = self.ewc.before_step_snapshot()
        except Exception:
            param_before = None

        # Step adapter optimizer if available (help adapters adapt continuously).
        if hasattr(self, 'adapter_optimizer') and self.adapter_optimizer is not None:
            try:
                self.adapter_optimizer.step()
                # Post-update: Clip adapter parameter norms to avoid catastrophic jumps
                try:
                    maxn = getattr(self.config, 'adapter_max_norm', None) or (self.config.gradient_clip_norm * 2.0)
                    if hasattr(self, 'adapter_bank') and self.adapter_bank is not None:
                        for p in self.adapter_bank.parameters():
                            # Some adapter params may not be Parameters; guard conversion
                            try:
                                if p is None: continue
                                with torch.no_grad():
                                    n = float(p.data.norm())
                                    if n > maxn and n > 0:
                                        p.data.mul_(maxn / (n + 1e-6))
                            except Exception:
                                pass
                except Exception:
                    # Adapter clipping must not break training loop
                    pass
            except Exception as e:
                self.logger.warning(f"Adapter optimizer step failed: {e}")

        if hasattr(self, 'adapter_optimizer') and self.adapter_optimizer is not None and (hasattr(self, 'adapter_only') and adapter_only):
            # Skip backbone update during adapter-only emergency
            pass
        else:
            # Apply Plasticity Gate (The Shield)
            if plasticity_gate < 0.99:
                if plasticity_gate < 0.01:
                    # Optimization: Skip update entirely if bored
                    if len(self.meta_log_probs) > 0: self.meta_log_probs.pop()
                    self.loss_history.append(current_mse_val) 
                    return {"loss": current_loss_val, "status": "skipped", "plasticity": 0.0}
                      
                with torch.no_grad():
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad.mul_(plasticity_gate)

            # Clip backbone gradients before optimizer step
            try:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            except Exception:
                pass
            self.optimizer.step()

            # If importance handler supports path accumulation (SI), update it now
            try:
                if param_before is not None and hasattr(self.ewc, 'accumulate_path'):
                    # SIHandler uses .accumulate_path(param_before)
                    self.ewc.accumulate_path(param_before)
            except Exception:
                try:
                    if param_before is not None and hasattr(self.ewc, 'accumulate'):
                        self.ewc.accumulate(param_before)
                except Exception:
                    pass

        # Post-update trace: record post-update norms
        if do_trace and trace_entry is not None:
            try:
                total_param_norm2 = 0.0
                for p in self.model.parameters():
                    try:
                        total_param_norm2 += float(p.data.norm().item() ** 2)
                    except Exception:
                        pass
                total_param_norm2 = float(total_param_norm2 ** 0.5)
                trace_entry['post_param_norm'] = total_param_norm2

                # adapter post-norms
                if hasattr(self, 'adapter_bank') and self.adapter_bank is not None:
                    an = []
                    try:
                        for p in self.adapter_bank.parameters():
                            try:
                                an.append(float(p.data.norm().item()))
                            except Exception:
                                an.append(None)
                    except Exception:
                        an = None
                    trace_entry['post_adapter_norms'] = an
            except Exception:
                pass
            # Append to persistent step_trace for checkpointing
            try:
                if getattr(self, 'step_trace', None) is None:
                    self.step_trace = []
                self.step_trace.append(trace_entry)
                maxr = getattr(self.config, 'trace_max_records', 1000)
                if len(self.step_trace) > maxr:
                    self.step_trace = self.step_trace[-maxr:]
            except Exception:
                pass
            # Also append a line-based JSON debug file for immediate inspection
            try:
                import json
                dbg_dir = Path('debug')
                dbg_dir.mkdir(parents=True, exist_ok=True)
                seed = os.environ.get('MM_SEED', None) or str(int(time.time()))
                dbg_file = dbg_dir / f'trace_stream_{seed}.ndjson'
                # Convert non-serializable entries to repr strings
                def safe(v):
                    try:
                        json.dumps(v)
                        return v
                    except Exception:
                        try:
                            return str(v)
                        except Exception:
                            return None

                serial = {k: safe(v) for k, v in trace_entry.items()}
                with open(dbg_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(serial) + '\n')
            except Exception:
                pass
        
        # RL Update (Introspection) 
        # Unclamp the "Scream" in Panic/Survival
        if self.reward_baseline == 0.0: self.reward_baseline = current_loss_val
        advantage = self.reward_baseline - current_loss_val
        
        # Only clamp signal if we are calm
        if mode in ["NORMAL", "NOVELTY"]:
             advantage = torch.clamp(torch.tensor(advantage), min=-1.0, max=1.0).item()
        
        self.reward_baseline = (1 - self.alpha) * self.reward_baseline + self.alpha * current_loss_val
        
        if meta_step and len(self.meta_log_probs) > 0:
            scale = 50.0 if mode in ["PANIC", "SURVIVAL"] else 10.0
            log_prob = self.meta_log_probs[-1]
            policy_loss = -log_prob * (advantage * scale)
            policy_loss.backward()
            try:
                # Clip meta (introspection) gradients to stabilize policy updates
                try:
                    torch.nn.utils.clip_grad_norm_(self.introspection_engine.parameters(), self.config.gradient_clip_norm)
                except Exception:
                    pass
                self.meta_optimizer.step()
            except Exception:
                pass
            self.meta_log_probs.clear()
        
        # Maintenance
        # CRITICAL FIX: Block Reptile during Panic
        if meta_step and not block_reptile:
            # Only run Reptile/meta-controller adapt if this is a real external step
            self.meta_controller.adapt(loss=current_loss_val)
        
        # Weight Editing
        if self.step_count % self.config.evaluation_frequency == 0:
            avg_loss = np.mean(self.loss_history) if self.loss_history else loss.item()
            internals = {'affine_modifiers': affine_modifiers, 'telemetry_buffer': self.telemetry_buffer, 'layer_map': self.layer_map}
            # Allow experiments to disable online weight editing (useful for short-run stability)
            if os.environ.get('MM_DISABLE_WEIGHT_EDIT', '0') != '1':
                try:
                    self.monitor.adapt_weights(current_loss=loss.item(), previous_loss=avg_loss, activations=internals)
                except Exception as e:
                    self.logger.warning(f"Weight editing failed (guarded): {e}")
            else:
                self.logger.info("MM_DISABLE_WEIGHT_EDIT=1 -> Skipping monitor.adapt_weights for stability")

        # Store MSE in history
        self.loss_history.append(current_mse_val)

        # Add to regular feedback buffer
        self.feedback_buffer.add(input_data, pred, target_data, -current_mse_val, current_mse_val)
        
        # Add to prioritized buffer if enabled (SOTA V7.0)
        # Include consciousness-derived importance score if available
        if hasattr(self, 'prioritized_buffer') and self.prioritized_buffer is not None:
            try:
                # Get the last snapshot from feedback buffer (just added)
                snapshot = self.feedback_buffer.buffer[-1] if self.feedback_buffer.buffer else None
                if snapshot is not None:
                    # Combine z-score surprise with consciousness importance
                    combined_importance = (abs(z_score) + cons_importance) / 2.0
                    self.prioritized_buffer.add(snapshot, z_score=z_score, importance=combined_importance)
            except Exception as e:
                self.logger.debug(f"Failed to add to prioritized buffer: {e}")
        
        # Automatic Dreaming (Replay Consolidation)
        replay_loss = 0.0
        # Respect explicit environment override to disable dreaming during diagnostics
        disable_dream_env = os.environ.get('MM_DISABLE_DREAM', '0')
        dream_int = getattr(self.config, 'dream_interval', 10)
        if enable_dream and getattr(self.config, 'enable_dreaming', True) and disable_dream_env != '1' and self.step_count > 0 and (dream_int > 0 and (self.step_count % dream_int == 0)):
             # Disable dreaming during panic (focus on reality)
             if mode not in ["PANIC", "SURVIVAL", "BOOTSTRAP"]:
                 dream_metrics = self.learn_from_buffer(batch_size=16, num_epochs=1)
                 replay_loss = dream_metrics.get('replay_loss', 0.0)
        
        if enable_dream: self.step_count += 1
        
        return {
            "loss": loss.item(),
            "mse": current_mse_val,
            "plasticity": plasticity_gate,
            "z_score": float(z_score) if isinstance(z_score, (float, int)) else z_score.item(),
            "mode": mode
        }
    
    def learn_from_buffer(self, batch_size: int = 32, num_epochs: int = 1) -> Dict[str, float]:
        """
        Active Replay ("Dreaming"): Re-trains on past experiences to consolidate memory.
        
        Supports both uniform and prioritized sampling (SOTA V7.0):
        - Uniform: Standard experience replay
        - Prioritized: Weight by loss/surprise/recency to emphasize hard examples
        """
        # 1. Safety Check: Do we have enough memories?
        if len(self.feedback_buffer.buffer) < batch_size:
            # If buffer is small, just learn from what we have
            if len(self.feedback_buffer.buffer) < 10:
                return {} # Too few to be useful
            batch_size = len(self.feedback_buffer.buffer)

        use_prioritized = getattr(self.config, 'use_prioritized_replay', True) and hasattr(self, 'prioritized_buffer') and self.prioritized_buffer is not None
        
        self.logger.info(
            f"[DREAM] Dreaming: Consolidating {num_epochs} epochs from "
            f"{len(self.feedback_buffer.buffer)} memories (sampling={'prioritized' if use_prioritized else 'uniform'})..."
        )
        
        replay_losses = []
        
        # 2. Optimization Loop
        self.model.train()
        
        for epoch in range(num_epochs):
            # Sample with or without prioritization
            if use_prioritized:
                # Use prioritized replay (emphasizes hard/surprising examples)
                samples = self.prioritized_buffer.sample_batch(
                    batch_size=batch_size,
                    use_priorities=True
                )
            else:
                # Uniform random sampling (classic experience replay)
                samples = random.sample(self.feedback_buffer.buffer, batch_size)
            
            # Safety check: Skip if samples is empty or invalid
            if not samples or len(samples) == 0:
                self.logger.warning("No samples returned from replay buffer, skipping epoch")
                continue
            
            # Filter out samples with invalid tensors
            valid_samples = [s for s in samples if s.input_data is not None and s.target is not None]
            if not valid_samples:
                self.logger.warning("No valid samples in replay batch, skipping epoch")
                continue
            
            # 3. Reconstruct Tensors & Move to Device
            # FIX: Using torch.cat (as you correctly have) to handle variable sizes
            try:
                inputs = torch.cat([s.input_data for s in valid_samples], dim=0).to(self.device)
                targets = torch.cat([s.target for s in valid_samples], dim=0).to(self.device)
            except RuntimeError as e:
                self.logger.warning(f"Failed to concatenate tensors in replay: {e}, skipping epoch")
                continue
            
            # 4. The Sync Step: Call train_step()
            # CRITICAL FIX: We must pass enable_dream=False here.
            # Additionally, disable meta updates for replay steps to avoid
            # polluting the meta-controller with synthetic replay losses.
            metrics = self.train_step(inputs, targets, enable_dream=False, meta_step=False)
            
            replay_losses.append(metrics['loss'])
            
        avg_loss = np.mean(replay_losses) if replay_losses else 0.0
        self.logger.info(f"[DREAM] Dream Complete. Avg Replay Loss: {avg_loss:.4f}")
        
        return {'replay_loss': avg_loss}

    def save_checkpoint(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state_dict = self.model.state_dict()
        if hasattr(self.model, '_orig_mod'): state_dict = self.model._orig_mod.state_dict()
             
        ewc_state = self.ewc.fisher_dict if hasattr(self.ewc, 'fisher_dict') else None
        
        # Prepare serializable step_trace (if present)
        serial_trace = None
        try:
            if getattr(self, 'step_trace', None) is not None:
                serial_trace = []
                for entry in self.step_trace:
                    try:
                        # Ensure all tensors are converted to python types
                        e = {}
                        for k, v in entry.items():
                            if hasattr(v, 'tolist'):
                                try:
                                    e[k] = v.tolist()
                                except Exception:
                                    e[k] = v
                            else:
                                e[k] = v
                        serial_trace.append(e)
                    except Exception:
                        serial_trace.append(entry)
        except Exception:
            serial_trace = None

        torch.save({
            'model_state': state_dict,
            'introspection_engine': self.introspection_engine.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'meta_optimizer': self.meta_optimizer.state_dict(),
            'ewc_fisher': ewc_state,
            'adapters': None if not hasattr(self, 'adapter_bank') else {str(k): {'scale': v['scale'].cpu(), 'shift': v['shift'].cpu()} for k, v in self.adapter_bank.adapters.items()},
            'config': self.config,
            'step_count': self.step_count
        ,
            'step_trace': serial_trace
        }, path)
        self.logger.info(f"Checkpoint saved to {path}")
        # Also dump trace separately for easier inspection (if tracing enabled)
        try:
            if serial_trace:
                dbg_dir = Path('debug')
                dbg_dir.mkdir(parents=True, exist_ok=True)
                seed = os.environ.get('MM_SEED', None) or str(int(time.time()))
                dbg_path = dbg_dir / f'trace_{seed}.npz'
                # Convert entries with nested lists to arrays where possible
                import numpy as _np
                # Save as object arrays to preserve structure
                _np.savez_compressed(str(dbg_path), trace=serial_trace)
                self.logger.info(f"    [TRACE] Saved trace bundle: {dbg_path}")
        except Exception:
            pass

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if hasattr(self.model, '_orig_mod'): self.model._orig_mod.load_state_dict(ckpt['model_state'])
        else: self.model.load_state_dict(ckpt['model_state'])
            
        if 'introspection_engine' in ckpt:
            self.introspection_engine.load_state_dict(ckpt['introspection_engine'])
        
        if 'ewc_fisher' in ckpt and ckpt['ewc_fisher'] is not None:
            self.ewc.fisher_dict = ckpt['ewc_fisher']

        # Load adapters if present
        if 'adapters' in ckpt and ckpt['adapters'] is not None and hasattr(self, 'adapter_bank'):
            for k, v in ckpt['adapters'].items():
                idx = int(k)
                self.adapter_bank.adapters[idx] = {'scale': torch.nn.Parameter(v['scale'].to(self.device)), 'shift': torch.nn.Parameter(v['shift'].to(self.device))}
            
        if 'meta_optimizer' in ckpt:
            self.meta_optimizer.load_state_dict(ckpt['meta_optimizer'])
            
        self.step_count = ckpt.get('step_count', 0)
        # 🚀 NEW: AUTO-TETHER
        # If we have an EWC handler, lock the loaded weights immediately.
        # This protects the "Pre-trained Brain" from instant forgetting.
        if hasattr(self, 'ewc'):
            self.ewc.lock_for_ttt(strength=500.0)
            self.logger.info(f"[SAFETY] Safety Tether auto-engaged (Strength: 500.0) for {path}")
            
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
    
    def apply_task_memory(self, name_or_path: str, blend: float = 1.0):
        """Load a saved task memory and apply its adapters/anchors to the framework.

        Returns the loaded payload for inspection.
        """
        try:
            payload = self.ewc.load_task_memory(name_or_path)
        except Exception as e:
            self.logger.error(f"Failed to load task memory: {e}")
            return None

        # Apply adapters if present
        if 'adapters' in payload and payload['adapters'] and hasattr(self, 'adapter_bank'):
            for k, v in payload['adapters'].items():
                idx = int(k)
                self.adapter_bank.adapters[idx] = {'scale': torch.nn.Parameter(v['scale'].to(self.device)), 'shift': torch.nn.Parameter(v['shift'].to(self.device))}
        # Optionally apply anchor to backbone
        if 'opt_param_dict' in payload and payload['opt_param_dict']:
            try:
                # apply anchor fully or blended
                for name, param in self.model.named_parameters():
                    if name in payload['opt_param_dict']:
                        anchor = payload['opt_param_dict'][name].to(param.device)
                        if blend >= 1.0:
                            param.data.copy_(anchor)
                        else:
                            param.data.mul_(1.0 - blend).add_(anchor * blend)
            except Exception as e:
                self.logger.warning(f"Failed to apply anchor params: {e}")

        return payload

    def auto_apply_best_task_memory(self, threshold: float = 0.8):
        """Search saved task memories by fingerprint and apply best match if above threshold.

        Returns True if an adapter/anchor was applied.
        """
        try:
            names = self.ewc.list_task_memories()
        except Exception:
            return False

        if not names:
            return False

        # Compute current fingerprint
        try:
            fp = self.telemetry_buffer.mean(dim=0)
            fp_vec = fp.flatten().float()
            fp_norm = fp_vec / (fp_vec.norm(p=2) + 1e-9)
        except Exception:
            return False

        best_name = None
        best_score = -1.0

        for name in names:
            try:
                payload = self.ewc.load_task_memory(name)
                meta = payload.get('meta', {})
                other_fp = meta.get('fingerprint', None)
                if other_fp is None:
                    continue
                other_vec = torch.tensor(other_fp, dtype=torch.float32, device=fp_vec.device)
                other_norm = other_vec / (other_vec.norm(p=2) + 1e-9)
                score = torch.nn.functional.cosine_similarity(fp_norm.unsqueeze(0), other_norm.unsqueeze(0)).item()
                if score > best_score:
                    best_score = score
                    best_name = name
            except Exception:
                continue

        if best_name and best_score >= threshold:
            self.logger.info(f"[MEMORY] Best task memory match: {best_name} (score={best_score:.3f}) — applying")
            # Apply adapters and optionally anchor
            self.apply_task_memory(best_name, blend=1.0)
            return True

        return False
             