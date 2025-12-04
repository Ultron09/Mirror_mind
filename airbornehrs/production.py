"""
Production Adapter: Simplified API for Inference & Online Learning
===================================================================

Provides easy-to-use interface for integrating adaptive meta-learning
into production systems with minimal boilerplate.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import logging

from .core import AdaptiveFramework, AdaptiveFrameworkConfig


class InferenceMode:
    """Enum for inference modes"""
    STATIC = "static"  # No online learning
    ONLINE = "online"  # Continuous learning from inference data
    BUFFERED = "buffered"  # Learn from batches of inference data


class ProductionAdapter:
    """
    Simplified interface for production deployment.
    
    Handles model loading, inference, and optional online learning
    with minimal configuration.
    
    Example:
        >>> adapter = ProductionAdapter.load_checkpoint("model.pt")
        >>> for batch in data_stream:
        ...     predictions = adapter.predict(batch, update=True)
    """
    
    def __init__(self, framework: AdaptiveFramework, inference_mode: str = InferenceMode.STATIC):
        """
        Initialize adapter.
        
        Args:
            framework: AdaptiveFramework instance
            inference_mode: One of {STATIC, ONLINE, BUFFERED}
        """
        self.framework = framework
        self.inference_mode = inference_mode
        self.logger = logging.getLogger('ProductionAdapter')
        
        # For buffered mode
        self.inference_buffer = []
        self.buffer_size = 32
        
        self.logger.info(f"ProductionAdapter initialized (mode: {inference_mode})")
    
    def predict(self,
                input_data: torch.Tensor,
                update: bool = False,
                target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Run inference on input data.
        
        Args:
            input_data: Input tensor (batch_size, ...)
            update: If True and target is provided, perform online learning
            target: Optional target for online learning
            
        Returns:
            Model output
        """
        output, uncertainty = self.framework.forward(input_data)
        
        if update and target is not None:
            self._update_online(input_data, output, target)
        
        return output
    
    def _update_online(self,
                      input_data: torch.Tensor,
                      output: torch.Tensor,
                      target: torch.Tensor):
        """
        Perform online learning update.
        
        Args:
            input_data: Input batch
            output: Model output
            target: Target batch
        """
        if self.inference_mode == InferenceMode.STATIC:
            self.logger.debug("Static mode: skipping update")
            return
        
        if self.inference_mode == InferenceMode.ONLINE:
            # Immediate update
            self.framework.model.train()
            metrics = self.framework.train_step(input_data, target)
            self.framework.model.eval()
            self.logger.debug(f"Online update: loss = {metrics['loss']:.4f}")
        
        elif self.inference_mode == InferenceMode.BUFFERED:
            # Buffer update
            self.inference_buffer.append((input_data.cpu(), target.cpu()))
            
            if len(self.inference_buffer) >= self.buffer_size:
                self._flush_buffer()
    
    def _flush_buffer(self):
        """Process buffered data"""
        if not self.inference_buffer:
            return
        
        inputs = torch.cat([item[0] for item in self.inference_buffer], dim=0)
        targets = torch.cat([item[1] for item in self.inference_buffer], dim=0)
        
        self.framework.model.train()
        metrics = self.framework.train_step(inputs, targets)
        self.framework.model.eval()
        
        self.logger.info(f"Batch update ({len(self.inference_buffer)} items): loss = {metrics['loss']:.4f}")
        
        self.inference_buffer = []
    
    def get_uncertainty(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Get uncertainty estimates for predictions.
        
        Returns:
            Uncertainty tensor (same shape as output)
        """
        _, uncertainty = self.framework.forward(input_data)
        return uncertainty
    
    def save_checkpoint(self, path: str):
        """Save production checkpoint"""
        self.framework.save_checkpoint(path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    @classmethod
    def load_checkpoint(cls,
                       path: str,
                       inference_mode: str = InferenceMode.STATIC,
                       device: Optional[str] = None) -> 'ProductionAdapter':
        """
        Load checkpoint and create adapter.
        
        Args:
            path: Path to checkpoint file
            inference_mode: Inference mode to use
            device: Device to load on (cuda/cpu)
            
        Returns:
            ProductionAdapter instance
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        
        # Create framework with default config
        config = AdaptiveFrameworkConfig()
        framework = AdaptiveFramework(config, device=device)
        
        # Load checkpoint
        framework.load_checkpoint(path)
        framework.model.eval()
        
        # Create and return adapter
        adapter = cls(framework, inference_mode)
        
        return adapter
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.framework.get_metrics()


# ==================== QUICK START EXAMPLES ====================

def example_training():
    """Example: Train with meta-controller"""
    print("Example: Training with adaptive meta-learning")
    
    config = AdaptiveFrameworkConfig(
        model_dim=128,
        num_layers=4,
        num_heads=4,
        batch_size=16
    )
    
    framework = AdaptiveFramework(config)
    
    # Dummy training data
    X_train = torch.randn(100, 10, 128)
    y_train = torch.randn(100, 10, 128)
    
    print("Training...")
    for epoch in range(3):
        framework.model.train()
        metrics = framework.train_step(X_train, y_train)
        print(f"  Epoch {epoch}: Loss = {metrics['loss']:.4f}")
    
    # Save checkpoint
    framework.save_checkpoint("model_checkpoint.pt")
    print("✅ Model saved")


def example_production():
    """Example: Inference with online learning"""
    print("\nExample: Production deployment with online learning")
    
    # Load model
    adapter = ProductionAdapter.load_checkpoint(
        "model_checkpoint.pt",
        inference_mode=InferenceMode.ONLINE
    )
    
    # Simulate inference stream
    for step in range(5):
        new_data = torch.randn(1, 10, 128)
        new_target = torch.randn(1, 10, 128)
        
        # Predict with online learning
        prediction = adapter.predict(new_data, update=True, target=new_target)
        uncertainty = adapter.get_uncertainty(new_data)
        
        print(f"  Step {step}: Uncertainty = {uncertainty.mean():.4f}")
    
    print("✅ Production inference complete")


if __name__ == "__main__":
    example_training()
    example_production()
