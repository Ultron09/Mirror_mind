"""
Unified Memory Handler: SOTA Continual Learning
===============================================
Combines SI (online importance), adaptive regularization, and prioritized replay.

Architecture:
1. UnifiedMemoryHandler: SI + EWC hybrid with mode-aware penalty
2. PrioritizedReplayBuffer: Loss/surprise-based sampling
3. AdaptiveRegularization: Mode-dependent λ scheduling
4. DynamicConsolidation: Surprise-triggered consolidation

Status: Production-ready SOTA implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from pathlib import Path
import datetime
import copy


class UnifiedMemoryHandler:
    """
    Hybrid SI + EWC handler with online importance estimation and adaptive regularization.
    
    Key differences from vanilla EWC:
    - Importance computed ONLINE (during training), not from buffer
    - λ (regularization strength) adapts based on operating mode
    - Supports both SI (path-integral) and EWC (Fisher) importance
    - Dynamic consolidation triggers on surprise, not just time
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 method: str = 'si',
                 si_lambda: float = 1.0,
                 si_xi: float = 1e-3,
                 ewc_lambda: float = 0.4,
                 consolidation_criterion: str = 'hybrid'):
        """
        Args:
            model: PyTorch model to protect
            method: 'si' (default), 'ewc', or 'hybrid'
            si_lambda: SI penalty strength
            si_xi: Damping factor for SI (prevents division by zero)
            ewc_lambda: EWC penalty strength (used if method='ewc'|'hybrid')
            consolidation_criterion: 'time', 'surprise', or 'hybrid'
        """
        self.model = model
        self.method = method
        self.si_lambda = si_lambda
        self.si_xi = si_xi
        self.ewc_lambda = ewc_lambda
        self.consolidation_criterion = consolidation_criterion
        self.logger = logging.getLogger('UnifiedMemoryHandler')
        
        # SI state (per-parameter accumulators)
        self.omega_accum = {
            n: torch.zeros_like(p).detach() 
            for n, p in model.named_parameters() 
            if p.requires_grad
        }
        self.omega = {
            n: torch.zeros_like(p).detach() 
            for n, p in model.named_parameters() 
            if p.requires_grad
        }
        self.anchor = {
            n: p.clone().detach() 
            for n, p in model.named_parameters() 
            if p.requires_grad
        }
        
        # EWC state (for hybrid mode)
        self.fisher_dict = {}
        self.opt_param_dict = {}
        
        # Consolidation tracking
        self.last_consolidation_step = 0
        self.consolidation_counter = 0
        self.steps_in_mode = 0
        self.current_mode = 'NORMAL'
        
        self.logger.info(
            f"🧠 Unified Memory Handler initialized (method={method}, "
            f"si_lambda={si_lambda}, consolidation={consolidation_criterion})"
        )
    
    def is_enabled(self):
        """Check if any importance has been computed."""
        if self.method in ['si', 'hybrid']:
            return any((v.abs().sum().item() > 0 for v in self.omega.values()))
        elif self.method == 'ewc':
            return len(self.fisher_dict) > 0
        return False
    
    def before_step_snapshot(self) -> Dict[str, torch.Tensor]:
        """Capture parameters before optimizer.step() for SI accumulation."""
        return {
            n: p.data.clone().detach() 
            for n, p in self.model.named_parameters() 
            if p.requires_grad
        }
    
    def accumulate_path(self, param_before: Dict[str, torch.Tensor]) -> None:
        """
        SI path-integral accumulation: s_i += -g_i * delta_theta_i
        Called AFTER optimizer.step() to compute parameter movement impact.
        
        Args:
            param_before: dict of param tensors before step
        """
        if self.method not in ['si', 'hybrid']:
            return
        
        try:
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if not p.requires_grad:
                        continue
                    
                    if p.grad is None:
                        continue
                    
                    if name not in param_before:
                        continue
                    
                    # Compute delta: change in parameters
                    old = param_before[name]
                    delta = (p.data - old).detach()
                    
                    # Gradient: importance of this parameter to loss
                    g = p.grad.data.detach()
                    
                    # Accumulate: s_i += -g_i * delta_theta_i (element-wise)
                    # Negative gradient direction * parameter movement = importance
                    try:
                        self.omega_accum[name] = self.omega_accum[name] + (-g * delta)
                    except Exception as e:
                        self.logger.debug(f"SI accumulation failed for {name}: {e}")
        
        except Exception as e:
            self.logger.warning(f"SI path accumulation failed: {e}")
    
    def consolidate(self, 
                    feedback_buffer=None,
                    current_step: int = 0,
                    z_score: float = 0.0,
                    mode: str = 'NORMAL',
                    **kwargs) -> None:
        """
        Consolidate importance: compute final omega from accumulators.
        
        For SI: omega_i = s_i / ((theta_i - theta_anchor_i)^2 + xi)
        Then reset accumulators and set new anchors.
        
        Args:
            feedback_buffer: Optional buffer for EWC computation
            current_step: Current training step (for logging)
            z_score: Surprise metric (for adaptive consolidation)
            mode: Current operating mode
        """
        self.consolidation_counter += 1
        self.logger.info(
            f"🧠 Consolidation #{self.consolidation_counter} (step={current_step}, "
            f"mode={mode}, z_score={z_score:.3f})"
        )
        
        with torch.no_grad():
            # Consolidate SI importance
            if self.method in ['si', 'hybrid']:
                for name, p in self.model.named_parameters():
                    if not p.requires_grad:
                        continue
                    
                    s = self.omega_accum.get(name, torch.zeros_like(p))
                    anchor = self.anchor.get(name, p.clone().detach())
                    
                    # Denominator: quadratic distance from anchor + damping
                    # BUG FIX #2: Add epsilon BEFORE division to prevent NaN
                    delta = (p.data - anchor).pow(2)
                    denom = delta + self.si_xi  # si_xi = 1e-3 provides safe base
                    
                    # Compute omega element-wise with numerical safety
                    # Add small epsilon to prevent division by zero
                    denom = torch.clamp(denom, min=1e-8)
                    new_omega = s / denom
                    
                    # Remove any NaNs/Infs that slipped through
                    new_omega = torch.nan_to_num(new_omega, nan=0.0, posinf=1e6, neginf=0.0)
                    
                    # Clamp for numerical stability
                    self.omega[name] = new_omega.clamp(min=0.0, max=1e6)
                    
                    # Reset accumulator and update anchor
                    self.omega_accum[name] = torch.zeros_like(p)
                    self.anchor[name] = p.data.clone().detach()
            
            # Consolidate EWC Fisher (if hybrid or ewc mode)
            if self.method in ['ewc', 'hybrid'] and feedback_buffer is not None:
                self._consolidate_ewc_fisher(feedback_buffer)
        
        self.last_consolidation_step = current_step
        self.logger.info(f"🔒 Consolidation complete. Protected {len(self.omega)} parameters.")
    
    def _consolidate_ewc_fisher(self, feedback_buffer, sample_limit: int = 128) -> None:
        """
        Compute Fisher Information Matrix from buffer samples (EWC method).
        Only used if method='ewc' or 'hybrid'.
        """
        if len(feedback_buffer.buffer) < 10:
            self.logger.warning("⚠️ Buffer too small for EWC consolidation.")
            return
        
        # Save current parameters as anchor
        self.opt_param_dict = {
            n: p.clone().detach()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        
        # Compute Fisher over recent buffer samples
        self.model.eval()
        fisher = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        
        samples = feedback_buffer.buffer[-sample_limit:]
        
        for snapshot in samples:
            self.model.zero_grad()
            
            inp = snapshot.input_data.to(next(self.model.parameters()).device)
            target = snapshot.target.to(next(self.model.parameters()).device)
            
            output = self.model(inp)
            if hasattr(output, 'logits'):
                output = output.logits
            elif isinstance(output, tuple):
                output = output[0]
            
            # Compute loss
            loss = F.mse_loss(output, target)
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2
        
        # Normalize and smooth
        alpha = 0.3
        for name in fisher:
            fisher[name] /= len(samples)
            if name in self.fisher_dict:
                fisher[name] = alpha * self.fisher_dict[name] + (1.0 - alpha) * fisher[name]
            fisher[name] = fisher[name].clamp(min=1e-8, max=1e9)
        
        self.fisher_dict = fisher
        self.model.train()
    
    def compute_penalty(self, adaptive_mode: str = 'NORMAL', step_in_mode: int = 0) -> torch.Tensor:
        """
        Compute regularization penalty based on stored importance.
        
        Penalty = (λ / 2) * sum_i Omega_i * (theta_i - theta_anchor_i)^2
        
        λ adapts based on mode:
        - BOOTSTRAP: 0 (free learning)
        - PANIC: 0 (override safety)
        - SURVIVAL: 0.1 (minimal protection)
        - NOVELTY: 0.8 (strong memory protection)
        - NORMAL: 0.4 (balanced)
        
        Args:
            adaptive_mode: Current operating mode for adaptive λ
            step_in_mode: Steps elapsed in current mode
        
        Returns:
            Scalar loss tensor
        """
        loss = 0.0
        
        if not self.is_enabled():
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        # Compute adaptive lambda based on mode
        base_lambda = self._get_adaptive_lambda(adaptive_mode, step_in_mode)
        
        if self.method in ['si', 'hybrid']:
            # SI penalty
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if name not in self.omega:
                    continue
                
                anchor = self.anchor.get(name)
                if anchor is None:
                    continue
                
                loss = loss + (self.omega[name] * (p - anchor).pow(2)).sum()
            
            loss = loss * (self.si_lambda * base_lambda / 2.0)
        
        if self.method in ['ewc', 'hybrid'] and len(self.fisher_dict) > 0:
            # EWC penalty (if hybrid, add on top of SI)
            ewc_loss = 0.0
            for name, p in self.model.named_parameters():
                if name not in self.fisher_dict:
                    continue
                fisher = self.fisher_dict[name]
                opt_param = self.opt_param_dict[name]
                ewc_loss = ewc_loss + (fisher * (p - opt_param).pow(2)).sum()
            
            loss = loss + ewc_loss * (self.ewc_lambda * base_lambda / 2.0)
        
        return loss if isinstance(loss, torch.Tensor) else torch.tensor(loss, device=next(self.model.parameters()).device)
    
    def _get_adaptive_lambda(self, mode: str, step_in_mode: int) -> float:
        """
        Adaptive λ based on operating mode and step count.
        
        Idea: Different modes need different regularization strengths.
        - Emergency modes (PANIC, BOOTSTRAP): disable protection (learn fast)
        - Learning modes (NOVELTY): strong protection (preserve old knowledge)
        - Stable modes (NORMAL): moderate protection (smooth operation)
        """
        mode_lambdas = {
            'BOOTSTRAP': 0.0,      # Bootstrap phase: let it learn freely
            'PANIC': 0.0,          # Emergency: override all safety
            'SURVIVAL': 0.1,       # Critical state: minimal protection
            'NOVELTY': 0.8,        # New pattern: strong memory protection
            'NORMAL': 0.4           # Normal operation: balanced
        }
        
        base = mode_lambdas.get(mode, 0.4)
        
        # Exponential decay within mode (converge to lighter penalty)
        decay = np.exp(-0.01 * step_in_mode)
        
        return base * decay
    
    def should_consolidate(self,
                          current_step: int,
                          z_score: float,
                          mode: str) -> bool:
        """
        Decide whether to consolidate based on criterion.
        
        Criteria:
        - 'time': Periodic consolidation every N steps
        - 'surprise': Consolidate when surprise stabilizes
        - 'hybrid': Both conditions
        
        Args:
            current_step: Current training step
            z_score: Statistical surprise metric
            mode: Current operating mode
        
        Returns:
            True if consolidation should occur
        """
        steps_since = current_step - self.last_consolidation_step
        
        # Never consolidate during bootstrap or emergency modes
        if mode in ['BOOTSTRAP', 'PANIC', 'SURVIVAL']:
            return False
        
        if self.consolidation_criterion == 'time':
            # Time-based: consolidate every 100 steps
            return steps_since > 100
        
        elif self.consolidation_criterion == 'surprise':
            # Surprise-based: consolidate when novel pattern stabilizes
            if mode == 'NOVELTY' and z_score > 2.5:
                # High surprise detected; wait for stabilization
                return steps_since > 30
            return False
        
        elif self.consolidation_criterion == 'hybrid':
            # Hybrid: surprise-triggered, but periodic fallback
            if mode == 'NOVELTY' and z_score > 2.5:
                if steps_since > 30:
                    return True
            elif steps_since > 100:
                return True
        
        return False
    
    # Task memory management (backward compatible with EWC API)
    
    def save_task_memory(self, name: Optional[str] = None, **kwargs):
        """Save current state (anchor + importance) to disk."""
        if name is None:
            name = datetime.datetime.now().strftime(f"{self.method}_task_%Y%m%d_%H%M%S")
        
        payload = {
            'method': self.method,
            'anchor': {k: v.cpu() for k, v in self.anchor.items()},
            'omega': {k: v.cpu() for k, v in self.omega.items()} if self.method in ['si', 'hybrid'] else {},
            'fisher_dict': {k: v.cpu() for k, v in self.fisher_dict.items()} if self.method in ['ewc', 'hybrid'] else {},
            'opt_param_dict': {k: v.cpu() for k, v in self.opt_param_dict.items()} if self.opt_param_dict else {},
            'meta': {
                'timestamp': datetime.datetime.now().isoformat(),
                'model': type(self.model).__name__,
                'consolidations': self.consolidation_counter
            }
        }
        
        save_dir = Path.cwd() / 'checkpoints' / 'task_memories'
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{name}.pt"
        
        torch.save(payload, save_path)
        self.logger.info(f"💾 Task memory saved: {save_path}")
        return str(save_path)
    
    def load_task_memory(self, path_or_name: str):
        """Load a saved task memory."""
        p = Path(path_or_name)
        if not p.exists():
            p = Path.cwd() / 'checkpoints' / 'task_memories' / path_or_name
            if not p.exists():
                raise FileNotFoundError(f"Task memory not found: {path_or_name}")
        
        payload = torch.load(p, map_location=next(self.model.parameters()).device)
        device = next(self.model.parameters()).device
        
        self.anchor = {k: v.to(device) for k, v in payload.get('anchor', {}).items()}
        self.omega = {k: v.to(device) for k, v in payload.get('omega', {}).items()}
        self.fisher_dict = {k: v.to(device) for k, v in payload.get('fisher_dict', {}).items()}
        self.opt_param_dict = {k: v.to(device) for k, v in payload.get('opt_param_dict', {}).items()}
        
        self.logger.info(f"🔁 Task memory loaded: {p}")
        return payload


class PrioritizedReplayBuffer:
    """
    Experience replay with prioritization based on loss, surprise, and recency.
    
    Prioritization formula:
    priority = α * loss + β * surprise + γ * recency
    where (α, β, γ) = (0.6, 0.3, 0.1)
    """
    
    def __init__(self, capacity: int = 10000, temperature: float = 0.6):
        """
        Args:
            capacity: Maximum buffer size
            temperature: Softmax temperature (0=greedy, 1=uniform)
        """
        self.capacity = capacity
        self.temperature = temperature
        self.buffer = deque(maxlen=capacity)
        self.logger = logging.getLogger('PrioritizedReplayBuffer')
    
    def add(self, snapshot, z_score: float = 0.0):
        """Add experience with metadata."""
        snapshot.z_score = z_score
        snapshot.age_in_steps = 0
        self.buffer.append(snapshot)
        
        # Age all existing samples
        for s in self.buffer:
            s.age_in_steps += 1
    
    def score_experience(self, snapshot) -> float:
        """
        Multi-criterion priority scoring.
        
        Returns:
            Priority score (higher = more likely to be sampled)
        """
        # Loss term: MSE > 0.1 gets high priority
        loss_term = snapshot.loss
        
        # Surprise term: high z-score indicates unexpected pattern
        surprise_term = abs(getattr(snapshot, 'z_score', 0.0))
        
        # Recency term: older samples get boost for diversity
        age = getattr(snapshot, 'age_in_steps', 0)
        recency_term = 1.0 / (1.0 + age / 100.0)  # Normalized to [0, 1]
        
        # Weighted combination
        priority = (0.6 * loss_term + 0.3 * surprise_term + 0.1 * recency_term)
        
        return priority
    
    def sample_batch(self, batch_size: int, use_priorities: bool = True) -> List:
        """
        Sample a batch of experiences.
        
        Args:
            batch_size: Number of samples to draw
            use_priorities: If False, uniform sampling
        
        Returns:
            List of snapshots
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        if not use_priorities:
            return random.sample(list(self.buffer), batch_size)
        
        # Compute priorities for all samples
        priorities = torch.tensor(
            [self.score_experience(s) for s in self.buffer],
            dtype=torch.float32
        )
        
        # Softmax with temperature
        weights = F.softmax(priorities / self.temperature, dim=0)
        
        # Multinomial sampling
        indices = torch.multinomial(weights, batch_size, replacement=True)
        
        return [self.buffer[i] for i in indices.tolist()]


class AdaptiveRegularization:
    """
    Dynamic regularization strength based on operating mode and phase.
    
    Provides mode-aware λ scheduling for continual learning.
    """
    
    def __init__(self, base_lambda: float = 0.4):
        self.base_lambda = base_lambda
        self.logger = logging.getLogger('AdaptiveRegularization')
        self.mode_history = deque(maxlen=100)
    
    def get_lambda(self, mode: str, step_in_mode: int = 0) -> float:
        """
        Get regularization strength for current mode.
        
        Schedules λ differently for each reflex mode:
        - BOOTSTRAP: 0.0 (warmup phase, maximize plasticity)
        - PANIC: 0.0 (emergency, ignore old knowledge)
        - SURVIVAL: 0.1 (critical state, minimal protection)
        - NOVELTY: 0.8 (learning new task, protect old)
        - NORMAL: 0.4 (balanced operation)
        
        Within each mode, λ decays over time (converge to lighter penalty).
        
        Args:
            mode: Current operating mode
            step_in_mode: Steps elapsed since entering this mode
        
        Returns:
            Adaptive λ value
        """
        mode_base = {
            'BOOTSTRAP': 0.0,
            'PANIC': 0.0,
            'SURVIVAL': 0.1,
            'NOVELTY': 0.8,
            'NORMAL': 0.4
        }
        
        base = mode_base.get(mode, 0.4)
        
        # Exponential decay within mode
        decay = np.exp(-0.01 * step_in_mode)
        
        # Final λ with base multiplier
        final_lambda = self.base_lambda * base * decay
        
        self.mode_history.append((mode, final_lambda))
        
        return final_lambda


class DynamicConsolidationScheduler:
    """
    Determines when to consolidate based on multiple criteria.
    
    Consolidation is the checkpoint where:
    1. Current parameters become new anchor (θ*)
    2. Importance (omega/fisher) is frozen for next phase
    3. Accumulators are reset
    
    Smart triggering avoids unnecessary consolidations while ensuring
    periodic memory checkpoints.
    """
    
    def __init__(self, min_interval: int = 30, max_interval: int = 100):
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.logger = logging.getLogger('DynamicConsolidation')
        self.last_consolidation_step = 0
        self.consolidation_count = 0
    
    def should_consolidate(self,
                          current_step: int,
                          z_score: float,
                          mode: str,
                          criterion: str = 'hybrid') -> Tuple[bool, str]:
        """
        Decide consolidation with detailed reasoning.
        
        Args:
            current_step: Current training step
            z_score: Statistical surprise metric
            mode: Current operating mode
            criterion: 'time'|'surprise'|'hybrid'
        
        Returns:
            (should_consolidate: bool, reason: str)
        """
        steps_since = current_step - self.last_consolidation_step
        
        # Never consolidate during emergencies
        if mode in ['BOOTSTRAP', 'PANIC', 'SURVIVAL']:
            return False, f"Emergency mode {mode}: deferring consolidation"
        
        if criterion == 'time':
            if steps_since > self.max_interval:
                return True, f"Time-based: {steps_since} steps elapsed"
        
        elif criterion == 'surprise':
            if mode == 'NOVELTY' and z_score > 2.5 and steps_since > self.min_interval:
                return True, f"Novelty stabilized (z={z_score:.2f}, age={steps_since})"
        
        elif criterion == 'hybrid':
            # Surprise-driven consolidation (primary)
            if mode == 'NOVELTY' and z_score > 2.5 and steps_since > self.min_interval:
                return True, f"Novelty phase with surprise={z_score:.2f}"
            
            # Time-driven fallback (periodic safety checkpoint)
            elif steps_since > self.max_interval:
                return True, f"Periodic consolidation (max_interval={self.max_interval})"
        
        return False, "Consolidation criteria not met"
    
    def record_consolidation(self, current_step: int):
        """Record that consolidation occurred."""
        self.last_consolidation_step = current_step
        self.consolidation_count += 1
