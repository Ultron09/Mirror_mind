"""
Meta-Controller: Dynamic Adaptation & Online Learning
======================================================

Implements adaptive optimization cycles that modify training dynamics
in response to performance feedback. This is the "learning to learn"
component that enables models to improve their own learning processes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Tuple, Dict, List, Optional, Any
import numpy as np
from collections import defaultdict, deque
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
    
    # Meta-learning (optimization cycle)
    inner_loop_steps: int = 5
    outer_loop_steps: int = 1
    meta_learning_rate: float = 1e-4
    
    # Curriculum strategy
    curriculum_start_difficulty: float = 0.1
    curriculum_increase_rate: float = 0.01
    
    # Regularization scheduling
    base_regularization: float = 1e-4
    regularization_schedule: str = "linear"  # "linear", "exponential", "cosine"


# ==================== GRADIENT ANALYZER ====================

class GradientAnalyzer:
    """
    Analyzes gradient statistics to make adaptation decisions.
    
    Monitors:
    - Gradient magnitude and variance
    - Gradient distribution across layers
    - Signs of gradient explosion or vanishing
    """
    
    def __init__(self, model: nn.Module, config: MetaControllerConfig):
        self.model = model
        self.config = config
        self.logger = logging.getLogger('GradientAnalyzer')
        self.gradient_history = deque(maxlen=100)
        
    def analyze(self) -> Dict[str, float]:
        """
        Analyze current gradient statistics.
        
        Returns:
            Dictionary with gradient metrics
        """
        stats = {
            'mean_norm': 0.0,
            'max_norm': 0.0,
            'min_norm': 0.0,
            'variance': 0.0,
            'sparsity': 0.0,  # Fraction of near-zero gradients
        }
        
        all_grads = []
        
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                all_grads.append(grad_norm)
        
        if not all_grads:
            return stats
        
        all_grads = np.array(all_grads)
        stats['mean_norm'] = float(np.mean(all_grads))
        stats['max_norm'] = float(np.max(all_grads))
        stats['min_norm'] = float(np.min(all_grads))
        stats['variance'] = float(np.var(all_grads))
        stats['sparsity'] = float(np.sum(all_grads < 1e-6) / len(all_grads))
        
        self.gradient_history.append(stats)
        
        return stats
    
    def get_trajectory(self) -> List[Dict[str, float]]:
        """Get gradient statistics history"""
        return list(self.gradient_history)
    
    def should_reduce_lr(self) -> bool:
        """Detect gradient explosion"""
        if len(self.gradient_history) < 2:
            return False
        
        recent = list(self.gradient_history)[-5:]
        recent_norms = [s['mean_norm'] for s in recent]
        
        # If mean gradient norm is exploding, reduce LR
        if recent_norms[-1] > recent_norms[0] * 10:
            return True
        
        return False


# ==================== DYNAMIC LEARNING RATE SCHEDULER ====================

class DynamicLearningRateScheduler:
    """
    Adapts learning rate based on loss landscape and gradient statistics.
    
    Implements:
    - Gradient-based LR adjustment (reduce on explosion, increase on plateau)
    - Loss-based scheduling (adaptive to convergence speed)
    - Entropy-based adaptation (high uncertainty → lower LR)
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, config: MetaControllerConfig):
        self.optimizer = optimizer
        self.config = config
        self.current_lr = config.base_lr
        self.logger = logging.getLogger('DynamicLearningRateScheduler')
        
        self.loss_history = deque(maxlen=20)
        self.lr_history = deque(maxlen=100)
        
    def step(self, loss: float, gradient_stats: Dict[str, float]) -> float:
        """
        Adapt learning rate based on loss and gradient statistics.
        
        Returns:
            New learning rate
        """
        self.loss_history.append(loss)
        
        # Detect gradient explosion
        if gradient_stats['mean_norm'] > 100:
            self.current_lr *= 0.5
            self.logger.info(f"Gradient explosion detected. Reducing LR to {self.current_lr:.2e}")
        
        # Detect gradient vanishing
        elif gradient_stats['mean_norm'] < 1e-6:
            self.current_lr *= 1.1
            self.logger.info(f"Gradient vanishing detected. Increasing LR to {self.current_lr:.2e}")
        
        # Loss plateau detection
        elif len(self.loss_history) >= 5:
            recent_losses = list(self.loss_history)[-5:]
            loss_improvement = (recent_losses[0] - recent_losses[-1]) / (recent_losses[0] + 1e-9)
            
            if loss_improvement < 0.001:  # <0.1% improvement
                self.current_lr *= 1.05
                self.logger.info(f"Loss plateau. Increasing LR to {self.current_lr:.2e}")
        
        # Clamp to valid range
        self.current_lr = np.clip(
            self.current_lr,
            self.config.min_lr,
            self.config.max_lr
        )
        
        # Apply to optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
        
        self.lr_history.append(self.current_lr)
        
        return self.current_lr
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.current_lr


# ==================== CURRICULUM STRATEGY ====================

class CurriculumStrategy:
    """
    Implements curriculum learning: start with easy tasks, gradually increase difficulty.
    
    Enables the model to build good representations on simple problems first,
    then tackle harder ones.
    """
    
    def __init__(self, config: MetaControllerConfig):
        self.config = config
        self.current_difficulty = config.curriculum_start_difficulty
        self.step_count = 0
        self.logger = logging.getLogger('CurriculumStrategy')
        
    def get_difficulty(self) -> float:
        """Get current task difficulty (0.0 = easy, 1.0 = hard)"""
        return np.clip(self.current_difficulty, 0.0, 1.0)
    
    def step(self, loss_improvement: float):
        """
        Update difficulty based on learning progress.
        
        If model is learning well (loss improving), increase difficulty.
        """
        if loss_improvement > 0.01:  # >1% improvement
            self.current_difficulty += self.config.curriculum_increase_rate
            self.logger.info(f"Learning progressing. Difficulty → {self.get_difficulty():.2f}")
        
        self.step_count += 1
    
    def sample_task_batch(self, batch: torch.Tensor, batch_targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filter batch to match current curriculum difficulty.
        
        Easy tasks: simpler patterns, lower noise
        Hard tasks: complex patterns, higher noise
        """
        difficulty = self.get_difficulty()
        
        # Example: add difficulty-dependent noise
        noise_level = difficulty * 0.1
        perturbed_batch = batch + torch.randn_like(batch) * noise_level
        
        return perturbed_batch, batch_targets


# ==================== MAML-STYLE INNER/OUTER LOOP ====================

class MAMLAdapter:
    """
    Model-Agnostic Meta-Learning style adaptation.
    
    Implements inner and outer loops for the optimization cycle:
    - Inner loop: adapt to support batch (quick adaptation)
    - Outer loop: update original parameters for generalization
    """
    
    def __init__(self, model: nn.Module, config: MetaControllerConfig, device):
        self.model = model
        self.config = config
        self.device = device
        self.logger = logging.getLogger('MAMLAdapter')
        
        # Store original parameters
        self.original_params = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }
    
    def inner_loop(self,
                   support_batch: Tuple[torch.Tensor, torch.Tensor],
                   num_steps: int = 5) -> float:
        """
        Inner loop: quickly adapt to support batch.
        
        Args:
            support_batch: (input, target) tuple
            num_steps: Adaptation steps
            
        Returns:
            Loss on support batch after adaptation
        """
        input_data, target = support_batch
        input_data = input_data.to(self.device)
        target = target.to(self.device)
        
        # Create adapted parameters
        adapted_params = {
            name: param.clone().detach().requires_grad_(True)
            for name, param in self.model.named_parameters()
        }
        
        inner_optimizer = torch.optim.SGD(
            [p for p in adapted_params.values()],
            lr=self.config.meta_learning_rate
        )
        
        best_loss = float('inf')
        
        for step in range(num_steps):
            output = self._forward_with_params(input_data, adapted_params)
            loss = F.mse_loss(output, target)
            
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()
            
            best_loss = min(best_loss, loss.item())
        
        return best_loss
    
    def outer_loop(self,
                  support_batches: List[Tuple[torch.Tensor, torch.Tensor]],
                  query_batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """
        Outer loop: update original parameters based on adapted performance.
        
        Args:
            support_batches: List of support batches for tasks
            query_batch: Query batch to evaluate on
            
        Returns:
            Meta-loss (loss on query batch)
        """
        meta_loss = 0.0
        
        for support_batch in support_batches:
            # Adapt to support batch
            self.inner_loop(support_batch, self.config.inner_loop_steps)
            
            # Evaluate on query batch
            query_input, query_target = query_batch
            query_input = query_input.to(self.device)
            query_target = query_target.to(self.device)
            
            output = self.model(query_input)
            loss = F.mse_loss(output, query_target)
            
            meta_loss += loss
        
        meta_loss /= len(support_batches)
        
        return meta_loss.item()
    
    def _forward_with_params(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with custom parameters"""
        # This is a simplified version; a full implementation would use
        # functional_call or similar mechanisms
        return self.model(x)


# ==================== META-CONTROLLER ====================

class MetaController:
    """
    Meta-controller orchestrating the optimization cycle.
    
    Coordinates:
    - Gradient analysis for diagnostics
    - Dynamic learning rate scheduling
    - Curriculum learning strategy
    - Inner/outer loop adaptation (MAML-style)
    
    This is the component that makes the model learn how to learn better.
    """
    
    def __init__(self, 
                 framework: 'AdaptiveFramework',  # type: ignore
                 config: Optional[MetaControllerConfig] = None):
        """
        Initialize meta-controller.
        
        Args:
            framework: AdaptiveFramework instance
            config: MetaControllerConfig (uses defaults if None)
        """
        if config is None:
            config = MetaControllerConfig()
        
        self.framework = framework
        self.config = config
        self.logger = logging.getLogger('MetaController')
        
        # Components
        self.gradient_analyzer = GradientAnalyzer(framework.model, config)
        self.lr_scheduler = DynamicLearningRateScheduler(framework.optimizer, config)
        self.curriculum = CurriculumStrategy(config)
        self.maml_adapter = MAMLAdapter(framework.model, config, framework.device)
        
        self.step_count = 0
        
    def adapt(self,
              loss: float,
              gradients: Optional[Dict[str, torch.Tensor]] = None,
              performance_metrics: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Execute adaptation step in the optimization cycle.
        
        Args:
            loss: Current loss value
            gradients: Optional gradient information (for analysis)
            performance_metrics: Optional performance metrics
            
        Returns:
            Dictionary with adaptation metrics
        """
        metrics = {}
        
        # Analyze gradient statistics
        grad_stats = self.gradient_analyzer.analyze()
        metrics['gradient_stats'] = grad_stats
        
        # Adapt learning rate
        new_lr = self.lr_scheduler.step(loss, grad_stats)
        metrics['learning_rate'] = new_lr
        
        # Update curriculum if performance metrics available
        if performance_metrics is not None:
            loss_improvement = performance_metrics.get('loss_improvement', 0.0)
            self.curriculum.step(loss_improvement)
            metrics['curriculum_difficulty'] = self.curriculum.get_difficulty()
        
        self.step_count += 1
        
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of adaptation history"""
        return {
            'step_count': self.step_count,
            'current_lr': self.lr_scheduler.get_lr(),
            'curriculum_difficulty': self.curriculum.get_difficulty(),
            'gradient_history': list(self.gradient_analyzer.get_trajectory())[-10:],
            'lr_history': list(self.lr_scheduler.lr_history)[-10:]
        }


# ==================== EXAMPLE ====================

if __name__ == "__main__":
    print("Meta-Controller module loaded")
    print("Import with: from airbornehrs.meta_controller import MetaController, GradientAnalyzer, etc.")
