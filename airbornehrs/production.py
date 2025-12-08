"""
Production Adapter: Simplified API for Inference & Online Learning
===================================================================

Provides a thread-safe, robust interface for integrating adaptive 
meta-learning into production systems.

Key Features:
- Seamless Meta-Controller Integration (Reptile/Dynamic LR)
- Thread-safe buffering for concurrent requests
- Optimized Inference Mode
- Automatic State Synchronization
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any, Union
from pathlib import Path
import logging
import threading

from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig
from airbornehrs.meta_controller import MetaController, MetaControllerConfig


class InferenceMode:
    """Enum for inference modes"""
    STATIC = "static"   # No online learning (Pure Inference)
    ONLINE = "online"   # Continuous learning (Immediate updates)
    BUFFERED = "buffered"  # Batched learning (High throughput)


class ProductionAdapter:
    """
    Production-ready wrapper for AdaptiveFramework.
    
    Integrates the MetaController to ensure online updates are safe,
    stable, and use the "Reptile" optimization strategy.
    """
    
    def __init__(self, 
                 framework: AdaptiveFramework, 
                 inference_mode: str = InferenceMode.STATIC,
                 enable_meta_learning: bool = True):
        """
        Initialize adapter.
        
        Args:
            framework: Initialized AdaptiveFramework
            inference_mode: Strategy for online updates
            enable_meta_learning: Whether to use MetaController (Reptile/LR Sched)
        """
        self.framework = framework
        self.inference_mode = inference_mode
        self.logger = logging.getLogger('ProductionAdapter')
        
        # THREAD SAFETY: Critical for production APIs
        self.lock = threading.Lock()
        
        # Buffer for batched updates
        self.inference_buffer = []
        self.buffer_size = 32
        
        # Initialize the "Brain" (Meta-Controller)
        # This handles the complexity of online adaptation (LR scaling, Reptile)
        if enable_meta_learning and inference_mode != InferenceMode.STATIC:
            self.meta_controller = MetaController(framework)
            self.logger.info("Meta-Controller attached for Online Adaptation.")
        else:
            self.meta_controller = None
            
        self.logger.info(f"ProductionAdapter initialized (mode: {inference_mode})")
    
    def predict(self,
                input_data: torch.Tensor,
                update: bool = False,
                target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Run inference on input data.
        
        Args:
            input_data: Input tensor
            update: If True, triggers online learning (requires target)
            target: Ground truth for adaptation
            
        Returns:
            Model prediction
        """
        # OPTIMIZATION: Use inference_mode when not updating
        # This disables gradient graph construction, saving RAM and Compute.
        if not update:
            with torch.inference_mode():
                output, _ = self.framework.forward(input_data)
            return output
        
        # If updating, we do a standard forward pass first to get the prediction
        # (We don't use this graph for training to keep logic clean)
        with torch.no_grad():
             output, _ = self.framework.forward(input_data)
        
        # Trigger Online Learning Logic
        if update and target is not None:
            self._update_online(input_data, target)
        
        return output
    
    def _update_online(self, input_data: torch.Tensor, target: torch.Tensor):
        """
        Handle online learning with Thread Safety and Meta-Control.
        """
        # 1. Static Mode: Do nothing
        if self.inference_mode == InferenceMode.STATIC:
            return
        
        # 2. Online Mode: Immediate Update
        if self.inference_mode == InferenceMode.ONLINE:
            with self.lock: # Ensure atomic updates
                self.framework.model.train()
                
                # A. Core Update (Gradients)
                metrics = self.framework.train_step(input_data, target)
                
                # B. Meta Update (Reptile/LR Schedule)
                if self.meta_controller:
                    self.meta_controller.adapt(
                        loss=metrics['loss'],
                        performance_metrics={'loss': metrics['loss']}
                    )
                    
                self.framework.model.eval()
                # self.logger.debug(f"Online update: loss={metrics['loss']:.4f}")
        
        # 3. Buffered Mode: Accumulate
        elif self.inference_mode == InferenceMode.BUFFERED:
            with self.lock:
                # Detach to save memory, move to CPU to free GPU for serving
                self.inference_buffer.append((input_data.detach().cpu(), target.detach().cpu()))
                
                if len(self.inference_buffer) >= self.buffer_size:
                    self._flush_buffer()
    
    def _flush_buffer(self):
        """
        Process accumulated buffer with Meta-Learning.
        Must be called within a lock or ensure single-threaded access.
        """
        if not self.inference_buffer:
            return
            
        # Move batch back to device
        inputs = torch.cat([item[0] for item in self.inference_buffer], dim=0).to(self.framework.device)
        targets = torch.cat([item[1] for item in self.inference_buffer], dim=0).to(self.framework.device)
        
        self.framework.model.train()
        
        # A. Core Update
        metrics = self.framework.train_step(inputs, targets)
        
        # B. Meta Update
        if self.meta_controller:
            self.meta_controller.adapt(
                loss=metrics['loss'],
                performance_metrics={'loss': metrics['loss']}
            )
        
        self.framework.model.eval()
        self.logger.info(f"Buffered update ({len(self.inference_buffer)} items): loss={metrics['loss']:.4f}")
        
        self.inference_buffer = []
    
    def get_uncertainty(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Get uncertainty estimates (Log Variance) for predictions.
        """
        with torch.inference_mode():
            _, log_var = self.framework.forward(input_data)
        return log_var
    
    def save_checkpoint(self, path: str):
        """Save production checkpoint"""
        # We rely on the framework's save mechanism
        self.framework.save_checkpoint(path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    @classmethod
    def load_checkpoint(cls,
                       path: str,
                       inference_mode: str = InferenceMode.STATIC,
                       device: Optional[str] = None) -> 'ProductionAdapter':
        """
        Robust checkpoint loading with state synchronization.
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)
            
        # 1. Load Checkpoint
        # weights_only=False required for dataclass config
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        # 2. Reconstruct Config
        config = checkpoint.get('config')
        if config is None:
            logging.warning("Config not found in checkpoint. Using defaults.")
            config = AdaptiveFrameworkConfig()
            
        # 3. Initialize Framework
        framework = AdaptiveFramework(config, device=device)
        
        # 4. Load Weights & Optimizer
        # Handle compiled model prefix if necessary
        model_state = checkpoint['model_state']
        if hasattr(framework.model, '_orig_mod'):
             # If current model is compiled but checkpoint wasn't (or vice versa), handle keys
             # This is a basic check; stricter matching might be needed for mixed envs
             pass 
             
        framework.model.load_state_dict(model_state)
        
        if 'optimizer_state' in checkpoint:
            framework.optimizer.load_state_dict(checkpoint['optimizer_state'])
            
        framework.step_count = checkpoint.get('step_count', 0)
        
        # 5. Create Adapter
        adapter = cls(framework, inference_mode)
        
        # 6. CRITICAL: Sync Meta-Controller with Loaded Optimizer
        # If we load a trained model, we must sync the LR scheduler
        if adapter.meta_controller:
            current_optim_lr = framework.optimizer.param_groups[0]['lr']
            adapter.meta_controller.lr_scheduler.current_lr = current_optim_lr
            logging.info(f"Synced Meta-Controller LR to: {current_optim_lr:.6f}")
        
        return adapter
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        metrics = self.framework.get_metrics()
        if self.meta_controller:
            metrics.update(self.meta_controller.get_summary())
        return metrics