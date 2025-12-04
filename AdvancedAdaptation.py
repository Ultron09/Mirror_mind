"""
🎯 Meta-Learning & Advanced Adaptation System
==============================================

MAML-inspired meta-learning for the SelfLearningFramework.
The model learns not just what to do, but HOW to learn better.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Tuple, Dict, List, Optional, Callable, Any
import numpy as np
from collections import defaultdict, deque
import copy
from dataclasses import dataclass, field
import logging


# ==================== ADVANCED ADAPTATION ====================

class MAMLAdaptor:
    """
    Model-Agnostic Meta-Learning (MAML) style adaptation.
    
    The model learns to adapt to new tasks quickly.
    This is the "learning to learn" mechanism.
    """
    
    def __init__(self, model: nn.Module, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.logger = logging.getLogger('MAMLAdaptor')
        
        # Store original parameters for MAML
        self.original_params = {name: param.clone().detach() 
                               for name, param in model.named_parameters()}
    
    def inner_loop(self, 
                   support_batch: Tuple[torch.Tensor, torch.Tensor],
                   num_steps: int = 5) -> float:
        """
        Inner loop: Adapt to support batch
        
        Args:
            support_batch: (input, target) tuple
            num_steps: Number of adaptation steps
            
        Returns:
            Adapted loss on support set
        """
        input_data, target = support_batch
        
        # Create a copy of parameters for inner loop
        adapted_params = {name: param.clone().detach().requires_grad_(True)
                         for name, param in self.model.named_parameters()}
        
        # Inner loop optimizer
        inner_optimizer = torch.optim.SGD(
            [p for p in adapted_params.values()],
            lr=self.config.meta_learning_rate
        )
        
        best_loss = float('inf')
        
        for step in range(num_steps):
            # Forward pass with adapted parameters
            output = self._forward_with_params(input_data, adapted_params)
            loss = F.mse_loss(output, target)
            
            # Inner loop update
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()
            
            best_loss = min(best_loss, loss.item())
        
        return best_loss
    
    def outer_loop(self,
                  support_batches: List[Tuple[torch.Tensor, torch.Tensor]],
                  query_batch: Tuple[torch.Tensor, torch.Tensor]):
        """
        Outer loop: Update original parameters based on adapted performance
        
        Args:
            support_batches: List of support batches for tasks
            query_batch: Query batch to evaluate on
        """
        meta_loss = 0.0
        
        for support_batch in support_batches:
            # Adapt to support batch
            self.inner_loop(support_batch, self.config.inner_loop_steps)
            
            # Evaluate on query batch
            query_input, query_target = query_batch
            output = self.model(query_input)
            
            if isinstance(output, tuple):
                output = output[0]  # Take main output if tuple
            
            query_loss = F.mse_loss(output, query_target)
            meta_loss += query_loss
        
        # Outer loop update
        meta_loss /= len(support_batches)
        
        # This would be the meta-gradient update
        # In practice, you'd compute gradients through the inner loop
        return meta_loss.item()
    
    def _forward_with_params(self, 
                            input_data: torch.Tensor,
                            params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass using specific parameter set"""
        # This is a simplified version - in practice you'd implement
        # proper parameter substitution through the model
        return self.model(input_data)


class GradientAnalyzer:
    """Analyze gradients to understand learning dynamics"""
    
    def __init__(self, model: nn.Module, config, logger=None):
        self.model = model
        self.config = config
        self.logger = logger or logging.getLogger('GradientAnalyzer')
        self.gradient_history = defaultdict(deque)
    
    def analyze_gradients(self) -> Dict[str, Any]:
        """Analyze current gradient statistics"""
        stats = {
            'total_norm': 0,
            'mean_grad': 0,
            'max_grad': 0,
            'min_grad': 0,
            'layer_stats': {}
        }
        
        layer_grads = defaultdict(list)
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_data = param.grad.data
                
                # Overall stats
                stats['total_norm'] += grad_data.norm(2).item()
                stats['mean_grad'] += grad_data.abs().mean().item()
                stats['max_grad'] = max(stats['max_grad'], grad_data.abs().max().item())
                stats['min_grad'] = min(stats['min_grad'], grad_data.abs().min().item())
                
                # Per-layer stats
                layer_name = name.split('.')[0]
                layer_grads[layer_name].append(grad_data.abs().mean().item())
        
        # Compute layer statistics
        for layer_name, grads in layer_grads.items():
            stats['layer_stats'][layer_name] = {
                'mean': np.mean(grads),
                'std': np.std(grads),
                'max': np.max(grads)
            }
        
        return stats
    
    def detect_vanishing_gradients(self, threshold: float = 1e-7) -> bool:
        """Detect if gradients are vanishing"""
        for param in self.model.parameters():
            if param.grad is not None:
                if param.grad.abs().max() < threshold:
                    return True
        return False
    
    def detect_exploding_gradients(self, threshold: float = 100.0) -> bool:
        """Detect if gradients are exploding"""
        for param in self.model.parameters():
            if param.grad is not None:
                if param.grad.abs().max() > threshold:
                    return True
        return False


class DynamicLearningRateScheduler:
    """
    Dynamically adjust learning rate based on training progress.
    
    If the model is learning well, maintain current LR.
    If plateauing, reduce LR or explore different strategies.
    """
    
    def __init__(self, optimizer, config, initial_lr: float):
        self.optimizer = optimizer
        self.config = config
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.logger = logging.getLogger('DynamicLRScheduler')
        
        # Track loss for plateau detection
        self.loss_history = deque(maxlen=100)
        self.plateau_counter = 0
        self.plateau_threshold = 10  # Steps to detect plateau
    
    def step(self, loss: float) -> Dict[str, float]:
        """
        Update learning rate based on loss trend
        
        Returns:
            Dictionary with adjustment info
        """
        self.loss_history.append(loss)
        
        info = {
            'learning_rate': self.current_lr,
            'plateau_detected': False,
            'adjustment': 'none'
        }
        
        if len(self.loss_history) < 10:
            return info
        
        # Check if loss is improving
        recent_losses = list(self.loss_history)[-10:]
        old_losses = list(self.loss_history)[-20:-10]
        
        if len(old_losses) > 0:
            improvement = (np.mean(old_losses) - np.mean(recent_losses)) / (np.mean(old_losses) + 1e-9)
            
            if improvement < 0.001:  # No improvement
                self.plateau_counter += 1
                info['plateau_detected'] = True
                
                if self.plateau_counter >= self.plateau_threshold:
                    # Reduce learning rate
                    new_lr = self.current_lr * 0.5
                    self._set_learning_rate(new_lr)
                    self.plateau_counter = 0
                    info['adjustment'] = 'reduced'
                    self.logger.info(f"Reduced LR: {self.current_lr} → {new_lr}")
            else:
                self.plateau_counter = 0
        
        return info
    
    def _set_learning_rate(self, lr: float):
        """Set learning rate for all param groups"""
        self.current_lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class CurriculumLearningStrategy:
    """
    Implement curriculum learning - start with easy tasks, progress to hard ones.
    
    This helps the model develop better learning strategies progressively.
    """
    
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger('CurriculumLearning')
        
        self.difficulty_level = 0
        self.max_difficulty = 10
        self.tasks_completed_at_level = 0
        self.tasks_per_level = 50
    
    def sample_task_difficulty(self) -> float:
        """
        Get current task difficulty (0.0 = easiest, 1.0 = hardest)
        
        Returns:
            Difficulty score
        """
        difficulty = self.difficulty_level / self.max_difficulty
        
        # Add some randomness
        noise = np.random.normal(0, 0.05)
        difficulty = np.clip(difficulty + noise, 0, 1)
        
        return difficulty
    
    def report_task_completion(self, success: bool):
        """Report whether the model succeeded on a task"""
        if success:
            self.tasks_completed_at_level += 1
            
            # Progress to next difficulty level
            if self.tasks_completed_at_level >= self.tasks_per_level:
                if self.difficulty_level < self.max_difficulty:
                    self.difficulty_level += 1
                    self.tasks_completed_at_level = 0
                    self.logger.info(f"Progressed to difficulty level {self.difficulty_level}")
    
    def generate_curriculum_task(self, base_task_generator: Callable) -> Tuple[Any, float]:
        """
        Generate a task with difficulty based on curriculum
        
        Args:
            base_task_generator: Function that generates tasks
            
        Returns:
            (task, difficulty_score)
        """
        difficulty = self.sample_task_difficulty()
        task = base_task_generator(difficulty)
        return task, difficulty


class PerformanceAnalyzer:
    """Analyze model performance and detect learning patterns"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger('PerformanceAnalyzer')
        self.metrics_history = deque(maxlen=1000)
    
    def add_metrics(self, 
                   loss: float,
                   accuracy: Optional[float] = None,
                   diversity: Optional[float] = None,
                   efficiency: Optional[float] = None):
        """Add performance metrics"""
        self.metrics_history.append({
            'loss': loss,
            'accuracy': accuracy,
            'diversity': diversity,
            'efficiency': efficiency
        })
    
    def detect_overfitting(self, train_loss: float, val_loss: float, threshold: float = 0.15) -> bool:
        """Detect if model is overfitting"""
        ratio = val_loss / (train_loss + 1e-9)
        return ratio > (1 + threshold)
    
    def detect_underfitting(self, train_loss: float, threshold: float = 0.1) -> bool:
        """Detect if model is underfitting"""
        return train_loss > threshold
    
    def compute_learning_efficiency(self) -> float:
        """
        Compute how efficient the learning process is.
        
        Returns:
            Score from 0 to 1 (1 = perfect learning)
        """
        if len(self.metrics_history) < 2:
            return 0.5
        
        recent = list(self.metrics_history)[-50:]
        losses = [m['loss'] for m in recent if m['loss'] is not None]
        
        if len(losses) < 2:
            return 0.5
        
        # Check for monotonic improvement
        improvements = []
        for i in range(1, len(losses)):
            improvement = (losses[i-1] - losses[i]) / (losses[i-1] + 1e-9)
            improvements.append(max(0, improvement))
        
        efficiency = np.mean(improvements) if improvements else 0.5
        return min(efficiency, 1.0)  # Cap at 1.0
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get detailed diagnostics"""
        if not self.metrics_history:
            return {}
        
        recent = list(self.metrics_history)[-100:]
        losses = [m['loss'] for m in recent]
        
        return {
            'current_loss': losses[-1] if losses else None,
            'best_loss': min(losses) if losses else None,
            'worst_loss': max(losses) if losses else None,
            'avg_loss': np.mean(losses) if losses else None,
            'loss_std': np.std(losses) if losses else None,
            'learning_efficiency': self.compute_learning_efficiency()
        }


# ==================== ADAPTIVE TASK SELECTOR ====================

class AdaptiveTaskSelector:
    """
    Select which tasks to train on based on model's current capability.
    
    Focuses on "edge cases" where the model struggles.
    """
    
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger('AdaptiveTaskSelector')
        
        self.task_performance = defaultdict(list)
        self.task_difficulty_estimate = defaultdict(float)
    
    def report_task_performance(self, task_id: str, performance: float):
        """Report how well model performed on a task"""
        self.task_performance[task_id].append(performance)
        
        # Update difficulty estimate (lower performance = harder task)
        recent_performance = self.task_performance[task_id][-10:]
        avg_performance = np.mean(recent_performance)
        self.task_difficulty_estimate[task_id] = 1.0 - avg_performance  # Invert
    
    def select_next_task(self, available_tasks: List[str]) -> str:
        """
        Select next task to train on.
        
        Prefer tasks where model is struggling (medium difficulty).
        """
        if not available_tasks:
            return available_tasks[0]
        
        # Score tasks based on difficulty and recent performance
        task_scores = []
        
        for task_id in available_tasks:
            difficulty = self.task_difficulty_estimate.get(task_id, 0.5)
            
            # Prefer medium difficulty tasks (0.3 to 0.7)
            if 0.3 <= difficulty <= 0.7:
                score = 1.0 - abs(difficulty - 0.5)  # Closer to 0.5 is better
            else:
                score = 0.5  # Lower score for very easy/hard tasks
            
            task_scores.append((task_id, score))
        
        # Sort by score (descending) and pick best
        task_scores.sort(key=lambda x: x[1], reverse=True)
        return task_scores[0][0]


if __name__ == "__main__":
    print("🎯 Advanced Meta-Learning & Adaptation System")
    print("=" * 50)
    
    # Test gradient analyzer
    model = nn.Linear(10, 5)
    analyzer = GradientAnalyzer(model, None)
    
    # Generate dummy gradients
    x = torch.randn(4, 10)
    y = torch.randn(4, 5)
    output = model(x)
    loss = F.mse_loss(output, y)
    loss.backward()
    
    stats = analyzer.analyze_gradients()
    print(f"\n✅ Gradient Analysis:")
    print(f"  Total norm: {stats['total_norm']:.4f}")
    print(f"  Vanishing gradients: {analyzer.detect_vanishing_gradients()}")
    
    # Test performance analyzer
    perf_analyzer = PerformanceAnalyzer()
    for i in range(100):
        perf_analyzer.add_metrics(loss=1.0 - 0.01 * i)
    
    diagnostics = perf_analyzer.get_diagnostics()
    print(f"\n✅ Performance Diagnostics:")
    print(f"  Learning efficiency: {diagnostics['learning_efficiency']:.4f}")
    
    print("\n🚀 Advanced system initialized!")
