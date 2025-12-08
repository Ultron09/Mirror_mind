"""
Meta-Controller: Reptile-Based Dynamic Adaptation
=================================================

Implements the "Reptile" meta-learning algorithm (OpenAI) adapted for
continuous online learning. This replaces brittle second-order MAML
with a stable "Lookahead" optimization strategy.

Algorithm:
1. Train normally for k steps (Fast Weights)
2. Update "Slow Weights" (Anchor) slightly towards Fast Weights
3. Reset Fast Weights to interpolated value
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Tuple, Dict, List, Optional, Any
import numpy as np
from collections import deque
import copy
from dataclasses import dataclass
import logging

# ==================== CONFIGURATION ====================

@dataclass
class MetaControllerConfig:
    """Configuration for the meta-controller"""
    # Learning rate scheduling
    base_lr: float = 1e-3
    min_lr: float = 1e-6
    max_lr: float = 1e-2
    
    # Gradient analysis
    gradient_clip_norm: float = 1.0
    
    # Reptile Meta-Learning (New Optimization Style)
    use_reptile: bool = True
    reptile_learning_rate: float = 0.1  # Epsilon (Interpolation rate)
    reptile_update_interval: int = 5    # k steps (Inner loop length)
    
    # Curriculum strategy
    curriculum_start_difficulty: float = 0.1
    curriculum_increase_rate: float = 0.01


# ==================== GRADIENT ANALYZER ====================

class GradientAnalyzer:
    """
    Analyzes gradient statistics to make adaptation decisions.
    """
    
    def __init__(self, model: nn.Module, config: MetaControllerConfig):
        self.model = model
        self.config = config
        self.logger = logging.getLogger('GradientAnalyzer')
        self.gradient_history = deque(maxlen=100)
        
    def analyze(self) -> Dict[str, float]:
        stats = {
            'mean_norm': 0.0,
            'max_norm': 0.0,
            'variance': 0.0,
            'sparsity': 0.0,
        }
        
        all_grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                if np.isfinite(grad_norm):
                    all_grads.append(grad_norm)
        
        if not all_grads:
            return stats
        
        all_grads = np.array(all_grads)
        stats['mean_norm'] = float(np.mean(all_grads))
        stats['max_norm'] = float(np.max(all_grads))
        stats['variance'] = float(np.var(all_grads))
        stats['sparsity'] = float(np.sum(all_grads < 1e-6) / len(all_grads))
        
        self.gradient_history.append(stats)
        return stats
    
    def get_trajectory(self) -> List[Dict[str, float]]:
        return list(self.gradient_history)


# ==================== ROBUST LR SCHEDULER ====================

class DynamicLearningRateScheduler:
    """
    Adapts learning rate with safety clamps for continuous learning.
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, config: MetaControllerConfig):
        self.optimizer = optimizer
        self.config = config
        self.current_lr = config.base_lr
        self.logger = logging.getLogger('DynamicLR')
        
        self.loss_history = deque(maxlen=20)
        self.lr_history = deque(maxlen=100)
        
    def step(self, loss: float, gradient_stats: Dict[str, float]) -> float:
        self.loss_history.append(loss)
        
        # 1. Gradient Explosion -> CUT LR
        if gradient_stats['mean_norm'] > 10.0:
            self.current_lr *= 0.5
            # self.logger.info("Explosion detected. Cutting LR.")
        
        # 2. Gradient Vanishing -> BOOST LR
        elif gradient_stats['mean_norm'] > 0 and gradient_stats['mean_norm'] < 1e-6:
            self.current_lr *= 1.1
            # self.logger.info("Vanishing detected. Boosting LR.")
        
        # 3. Loss Plateau (Robust Logic)
        elif len(self.loss_history) >= 5:
            recent = list(self.loss_history)[-5:]
            improvement = (recent[0] - recent[-1]) # Absolute difference
            
            if abs(improvement) < 0.01:
                self.current_lr *= 1.05 # Boost plasticity
            else:
                self.current_lr *= 0.95 # Stabilize
        
        # SAFETY: Prevent "Brain Death" (Zero LR)
        self.current_lr = np.clip(
            self.current_lr,
            max(self.config.min_lr, 1e-5), # Force minimum 1e-5
            self.config.max_lr
        )
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
        
        self.lr_history.append(self.current_lr)
        return self.current_lr
    
    def get_lr(self) -> float:
        return self.current_lr


# ==================== CURRICULUM STRATEGY ====================

class CurriculumStrategy:
    """
    Manages task difficulty with multi-modal safety checks.
    """
    
    def __init__(self, config: MetaControllerConfig):
        self.config = config
        self.current_difficulty = config.curriculum_start_difficulty
        self.logger = logging.getLogger('Curriculum')
        
    def get_difficulty(self) -> float:
        return np.clip(self.current_difficulty, 0.0, 1.0)
    
    def step(self, loss_improvement: float):
        if loss_improvement > 0.01:
            self.current_difficulty += self.config.curriculum_increase_rate
    
    def sample_task_batch(self, batch: torch.Tensor, batch_targets: torch.Tensor):
        difficulty = self.get_difficulty()
        
        # SAFETY CHECK: Only add noise to Floats (Images/Audio)
        # Prevents crash on NLP Integers
        if torch.is_floating_point(batch):
            noise_level = difficulty * 0.1
            perturbed_batch = batch + torch.randn_like(batch) * noise_level
            return perturbed_batch, batch_targets
        
        return batch, batch_targets


# ==================== REPTILE OPTIMIZER ====================

class ReptileOptimizer:
    """
    Implements the Reptile 'Lookahead' update rule.
    
    Logic:
    theta_new = theta_old + epsilon * (theta_fast - theta_old)
    
    This pulls the 'slow weights' towards the 'fast weights' learned
    over the last k steps, providing stability and generalization.
    """
    
    def __init__(self, model: nn.Module, config: MetaControllerConfig):
        self.model = model
        self.config = config
        self.anchor_weights = None
        self.step_counter = 0
        self.logger = logging.getLogger('ReptileOptimizer')
        
    def step(self):
        """
        Called every training step. Performs meta-update every k steps.
        """
        self.step_counter += 1
        
        # 1. Initialization: Set Anchor (Slow Weights)
        if self.anchor_weights is None:
            self.anchor_weights = self._clone_weights()
            return
            
        # 2. Check interval
        if self.step_counter % self.config.reptile_update_interval == 0:
            self._perform_update()
            
    def _clone_weights(self) -> Dict[str, torch.Tensor]:
        """Deep copy current model weights"""
        return {
            k: v.clone().detach() 
            for k, v in self.model.state_dict().items()
        }
        
    def _perform_update(self):
        """Performs the Reptile interpolation"""
        current_weights = self.model.state_dict()
        new_state_dict = {}
        
        epsilon = self.config.reptile_learning_rate
        
        with torch.no_grad():
            for name, anchor_param in self.anchor_weights.items():
                if name in current_weights:
                    fast_param = current_weights[name]
                    
                    # Reptile Update Rule:
                    # Anchor <- Anchor + epsilon * (Fast - Anchor)
                    new_param = anchor_param + epsilon * (fast_param - anchor_param)
                    
                    new_state_dict[name] = new_param
        
        # Apply updated weights to model
        self.model.load_state_dict(new_state_dict)
        
        # Update Anchor for next cycle
        self.anchor_weights = self._clone_weights()
        
        # self.logger.debug(f"Reptile update performed (Step {self.step_counter})")


# ==================== META-CONTROLLER ====================

class MetaController:
    """
    Orchestrates the optimization cycle using Reptile logic.
    """
    
    def __init__(self, 
                 framework: 'AdaptiveFramework', 
                 config: Optional[MetaControllerConfig] = None):
        if config is None:
            config = MetaControllerConfig()
        
        self.framework = framework
        self.config = config
        
        # Components
        self.gradient_analyzer = GradientAnalyzer(framework.model, config)
        self.lr_scheduler = DynamicLearningRateScheduler(framework.optimizer, config)
        self.curriculum = CurriculumStrategy(config)
        
        # Replaced MAML with Reptile
        self.reptile = ReptileOptimizer(framework.model, config)
        
        self.step_count = 0
        
    def adapt(self,
              loss: float,
              gradients: Optional[Dict[str, torch.Tensor]] = None,
              performance_metrics: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        
        metrics = {}
        
        # 1. Analyze Gradients
        grad_stats = self.gradient_analyzer.analyze()
        metrics['gradient_stats'] = grad_stats
        
        # 2. Schedule Learning Rate
        new_lr = self.lr_scheduler.step(loss, grad_stats)
        metrics['learning_rate'] = new_lr
        
        # 3. Update Curriculum
        if performance_metrics:
            loss_imp = performance_metrics.get('loss_improvement', 0.0)
            self.curriculum.step(loss_imp)
            metrics['curriculum_difficulty'] = self.curriculum.get_difficulty()
            
        # 4. REPTILE META-UPDATE
        if self.config.use_reptile:
            self.reptile.step()
        
        self.step_count += 1
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        return {
            'step_count': self.step_count,
            'current_lr': self.lr_scheduler.get_lr(),
            'reptile_active': self.config.use_reptile
        }

if __name__ == "__main__":
    print("Meta-Controller (Reptile Edition) loaded.")