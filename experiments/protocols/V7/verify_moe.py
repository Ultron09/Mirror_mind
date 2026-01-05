import torch
import torch.nn as nn
import sys
import os
import logging

# Path Setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MoE_Verify")

def run_verification():
    logger.info("🧪 Verifying Cortex Engine (MoE)...")
    
    # Define a simple base model
    base_model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    
    # Config with MoE enabled
    config = AdaptiveFrameworkConfig(
        use_moe=True,
        num_experts=4,
        top_k_experts=2,
        input_dim=10, # Required for gating
        device='cpu'
    )
    
    # Initialize Framework
    framework = AdaptiveFramework(base_model, config)
    
    # Check structure
    logger.info(f"Model Structure: {framework.model}")
    if not hasattr(framework.model, 'experts'):
        logger.error("❌ MoE Transformation Failed: 'experts' attribute missing.")
        return
        
    num_experts = len(framework.model.experts)
    logger.info(f"Number of Experts: {num_experts}")
    if num_experts != 4:
        logger.error(f"❌ Expected 4 experts, got {num_experts}")
        return
        
    # Test Forward Pass
    x = torch.randn(5, 10)
    y = torch.randn(5, 1)
    
    logger.info("Running Forward Pass...")
    output, _, _ = framework(x)
    
    # Handle tuple output if framework doesn't unpack it (it does unpack in forward, but returns tuple)
    # framework.forward returns (output, log_var, affine_modifiers)
    # output itself might be (tensor, indices) if framework didn't unpack it fully?
    # In core.py:
    # output = self.model(*args, **kwargs)
    # if isinstance(output, tuple) ... output, moe_indices = output
    # So 'output' variable in forward is the Tensor.
    
    logger.info(f"Output Shape: {output.shape}")
    if output.shape != (5, 1):
        logger.error(f"❌ Output shape mismatch. Expected (5, 1), got {output.shape}")
        return
        
    # Test Training Step
    logger.info("Running Train Step...")
    metrics = framework.train_step(x, target_data=y)
    logger.info(f"Metrics: {metrics}")
    
    logger.info("✅ MoE Verification Passed!")

if __name__ == "__main__":
    run_verification()
