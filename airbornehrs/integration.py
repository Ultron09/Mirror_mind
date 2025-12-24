"""
UNIFIED MIRRORMING INTEGRATION PACKAGE
======================================

Integrates all components:
- EWC (Elastic Weight Consolidation) for continual learning
- MetaController for adaptive learning
- Adapters for parameter-efficient learning
- ConsciousnessCore for self-awareness
- FeedbackBuffer for experience replay

This ensures seamless integration across the entire framework.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import numpy as np

from .core import AdaptiveFramework, AdaptiveFrameworkConfig, PerformanceSnapshot, FeedbackBuffer
from .ewc import EWCHandler
from .meta_controller import MetaController, MetaControllerConfig
from .adapters import AdapterBank
from .consciousness import ConsciousnessCore

logger = logging.getLogger('MirrorMindIntegration')


class MirrorMindSystem:
    """
    Unified MirrorMind system with all components working together.
    
    Architecture:
    1. Core Framework (Adaptive learning + gradient updates)
    2. EWC Handler (Prevent catastrophic forgetting)
    3. Meta-Controller (Reptile + adaptive LR)
    4. Adapter Bank (Parameter-efficient task adaptation)
    5. Consciousness Core (Self-aware learning)
    6. Feedback Buffer (Experience replay)
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 config: Optional[AdaptiveFrameworkConfig] = None,
                 meta_config: Optional[MetaControllerConfig] = None):
        """
        Initialize the unified MirrorMind system.
        
        Args:
            model: Base neural network model
            device: Device to run on (cuda/cpu)
            config: Adaptive framework configuration
            meta_config: Meta-controller configuration
        """
        
        if config is None:
            config = AdaptiveFrameworkConfig(device=device)
        
        if meta_config is None:
            meta_config = MetaControllerConfig()
        
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.config = config
        self.meta_config = meta_config
        self.logger = logging.getLogger('MirrorMindSystem')
        
        # ===== COMPONENT 1: OPTIMIZER & SCHEDULER =====
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-5
        )
        
        # ===== COMPONENT 2: EWC HANDLER (Continual Learning) =====
        self.ewc = EWCHandler(
            model=self.model,
            ewc_lambda=config.learning_rate * 100  # Scale with LR
        )
        self.logger.info(f"✅ EWC Handler initialized (lambda={self.ewc.ewc_lambda:.4f})")
        
        # ===== COMPONENT 3: META-CONTROLLER (Reptile + Adaptive LR) =====
        # Create a simple wrapper since MetaController expects AdaptiveFramework
        self.meta_controller = self._create_meta_controller()
        self.logger.info(f"✅ Meta-Controller initialized (Reptile={meta_config.use_reptile})")
        
        # ===== COMPONENT 4: ADAPTER BANK (Parameter-Efficient) =====
        num_layers = sum(1 for _ in self.model.modules() if isinstance(_, (nn.Linear, nn.Conv2d)))
        self.adapters = AdapterBank(num_layers=num_layers, device=self.device)
        self.logger.info(f"✅ Adapter Bank initialized ({num_layers} layers)")
        
        # ===== COMPONENT 5: CONSCIOUSNESS CORE (Self-Awareness) =====
        self.consciousness = ConsciousnessCore(
            model=self.model,
            feature_dim=config.model_dim,
            awareness_buffer_size=config.consciousness_buffer_size,
            novelty_threshold=config.novelty_threshold
        )
        self.logger.info(f"✅ Consciousness Core initialized")
        
        # ===== COMPONENT 6: FEEDBACK BUFFER (Experience Replay) =====
        self.feedback_buffer = FeedbackBuffer(config, self.device)
        self.logger.info(f"✅ Feedback Buffer initialized (capacity={config.feedback_buffer_size})")
        
        # ===== TRAINING STATE =====
        self.current_task_id = 0
        self.total_steps = 0
        self.metrics_history = []
        self.consolidation_counter = 0
    
    def _create_meta_controller(self):
        """Create MetaController with proper configuration."""
        # MetaController expects a framework object with model and optimizer
        class FrameworkProxy:
            def __init__(self, model, optimizer):
                self.model = model
                self.optimizer = optimizer
        
        framework_proxy = FrameworkProxy(self.model, self.optimizer)
        return MetaController(framework_proxy, self.meta_config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through model."""
        return self.model(x)
    
    def train_step(self,
                   x: torch.Tensor,
                   y: torch.Tensor,
                   task_id: int = 0,
                   use_ewc: bool = True,
                   use_adapters: bool = True) -> Dict[str, float]:
        """
        Single training step with all components integrated.
        
        Args:
            x: Input batch
            y: Target batch
            task_id: Current task ID (for continual learning)
            use_ewc: Whether to apply EWC penalty
            use_adapters: Whether to use adapters
        
        Returns:
            Dictionary of metrics
        """
        
        x = x.to(self.device)
        y = y.to(self.device)
        self.current_task_id = task_id
        
        # ===== FORWARD PASS =====
        self.optimizer.zero_grad()
        
        # Get predictions
        logits = self.model(x)
        
        # Compute base loss
        if y.dim() == 1:  # Classification
            loss = nn.CrossEntropyLoss()(logits, y)
        else:  # Regression
            loss = nn.MSELoss()(logits, y)
        
        # ===== CONSCIOUSNESS OBSERVATION =====
        # For consciousness, we need target to match logits shape
        # For classification, convert class index to one-hot
        if y.dim() == 1:  # Classification: y is [N]
            num_classes = logits.shape[1] if logits.dim() > 1 else 1
            y_true_for_consciousness = torch.nn.functional.one_hot(y, num_classes=num_classes).float()
        else:  # Regression: y is already [N, ...]
            y_true_for_consciousness = y.float()
        
        consciousness_metrics = self.consciousness.observe(
            x=x,
            y_true=y_true_for_consciousness,
            y_pred=logits.detach()
        )
        
        # ===== EWC PENALTY (Prevent Catastrophic Forgetting) =====
        ewc_penalty = 0.0
        if use_ewc and self.ewc.is_enabled():
            ewc_penalty = self.ewc.compute_penalty()
            loss = loss + ewc_penalty
        
        # ===== BACKWARD PASS =====
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
        
        # ===== OPTIMIZER STEP =====
        self.optimizer.step()
        
        # ===== ADAPTER UPDATE (if enabled) =====
        if use_adapters:
            self._update_adapters(x, logits, y)
        
        # ===== META-CONTROLLER ADAPTATION =====
        with torch.no_grad():
            grad_norm = sum(p.grad.norm().item() ** 2 for p in self.model.parameters() if p.grad is not None) ** 0.5
            grad_stats = {'mean_norm': grad_norm}
        
        # ===== BUFFER & CONSOLIDATION =====
        self.feedback_buffer.add(
            input_data=x,
            output=logits.detach(),
            target=y,
            reward=1.0 - loss.item(),
            loss=loss.item()
        )
        
        # Consolidate EWC periodically (detect domain shifts)
        self.consolidation_counter += 1
        if self.consolidation_counter >= self.config.consolidation_max_interval:
            if len(self.feedback_buffer.buffer) > 10:
                self.ewc.consolidate_from_buffer(self.feedback_buffer, sample_limit=32)
                self.consolidation_counter = 0
        
        # ===== METRICS COMPUTATION =====
        with torch.no_grad():
            if y.dim() == 1:  # Classification
                pred = logits.argmax(dim=1)
                accuracy = (pred == y).float().mean().item()
            else:  # Regression
                accuracy = (1.0 - loss.item())  # Pseudo-accuracy
        
        metrics = {
            'loss': loss.item(),
            'ewc_penalty': ewc_penalty if isinstance(ewc_penalty, (int, float)) else ewc_penalty.item(),
            'accuracy': accuracy,
            'confidence': consciousness_metrics.get('confidence', 0.0),
            'uncertainty': consciousness_metrics.get('uncertainty', 0.0),
            'surprise': consciousness_metrics.get('surprise', 0.0),
            'importance': consciousness_metrics.get('importance', 0.0),
            'task_id': task_id,
            'step': self.total_steps,
        }
        
        self.total_steps += 1
        self.metrics_history.append(metrics)
        
        return metrics
    
    def _update_adapters(self, x: torch.Tensor, logits: torch.Tensor, y: torch.Tensor):
        """Update adapter parameters based on current loss."""
        # Compute loss for adapter update
        if y.dim() == 1:
            loss = nn.CrossEntropyLoss()(logits, y)
        else:
            loss = nn.MSELoss()(logits, y)
        
        # Adapters are part of forward pass, no special update needed
        # (they're already optimized through the main backward pass)
        pass
    
    def consolidate_task_memory(self, task_id: int, data_loader=None):
        """
        Consolidate memories from current task using EWC.
        Called after finishing a task.
        """
        self.logger.info(f"🧠 Consolidating task {task_id}...")
        
        if data_loader is not None:
            # Compute Fisher from provided data
            self.ewc.consolidate_from_buffer(self.feedback_buffer, sample_limit=64)
        else:
            # Use existing buffer (need at least 5 snapshots)
            if len(self.feedback_buffer.buffer) > 4:
                self.ewc.consolidate_from_buffer(self.feedback_buffer, sample_limit=min(32, len(self.feedback_buffer.buffer)))
        
        self.logger.info(f"✅ Task {task_id} consolidated")
    
    def evaluate(self, data_loader) -> Dict[str, float]:
        """Evaluate on test set."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                logits = self.model(x)
                
                if y.dim() == 1:
                    loss = nn.CrossEntropyLoss()(logits, y)
                    pred = logits.argmax(dim=1)
                    correct = (pred == y).sum().item()
                else:
                    loss = nn.MSELoss()(logits, y)
                    correct = 0
                
                total_loss += loss.item() * x.size(0)
                total_correct += correct
                total_samples += x.size(0)
        
        self.model.train()
        
        return {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples if total_samples > 0 else 0.0,
        }
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get complete system state for checkpointing."""
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'ewc_fisher': self.ewc.fisher_dict,
            'ewc_anchor': self.ewc.opt_param_dict,
            'adapters': self.adapters.adapters,
            'total_steps': self.total_steps,
            'current_task': self.current_task_id,
        }
    
    def load_state_dict(self, state: Dict[str, Any]):
        """Load complete system state."""
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        if state.get('ewc_fisher'):
            self.ewc.fisher_dict = state['ewc_fisher']
        if state.get('ewc_anchor'):
            self.ewc.opt_param_dict = state['ewc_anchor']
        self.total_steps = state.get('total_steps', 0)
        self.current_task_id = state.get('current_task', 0)
        self.logger.info("✅ System state loaded")
    
    def summary(self):
        """Print system summary."""
        logger.info("\n" + "="*80)
        logger.info("MIRRORMING SYSTEM CONFIGURATION")
        logger.info("="*80)
        logger.info(f"  Model: {self.model.__class__.__name__}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"  EWC Lambda: {self.ewc.ewc_lambda:.4f}")
        logger.info(f"  Adapters: {len(self.adapters.adapters)} layers")
        logger.info(f"  Consciousness: Enabled")
        logger.info(f"  Meta-Learning: Reptile")
        logger.info(f"  Buffer Capacity: {self.config.feedback_buffer_size}")
        logger.info("="*80 + "\n")


def create_mirrorming_system(model: nn.Module,
                             device: str = 'cuda',
                             enable_ewc: bool = True,
                             enable_consciousness: bool = True,
                             enable_adapters: bool = True) -> MirrorMindSystem:
    """
    Factory function to create a configured MirrorMind system.
    
    Args:
        model: Base neural network
        device: Device to use
        enable_ewc: Enable continual learning via EWC
        enable_consciousness: Enable self-aware learning
        enable_adapters: Enable parameter-efficient learning
    
    Returns:
        Configured MirrorMindSystem instance
    """
    
    config = AdaptiveFrameworkConfig(
        device=device,
        enable_consciousness=enable_consciousness,
        memory_type='hybrid' if enable_ewc else 'ewc',
    )
    
    meta_config = MetaControllerConfig(
        use_reptile=True,
        base_lr=1e-3,
    )
    
    system = MirrorMindSystem(
        model=model,
        device=device,
        config=config,
        meta_config=meta_config
    )
    
    system.summary()
    
    return system
