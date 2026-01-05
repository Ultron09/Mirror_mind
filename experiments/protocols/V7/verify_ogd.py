import torch
import torch.nn as nn
import sys
import os
import logging
import numpy as np

# Path Setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from airbornehrs.memory import UnifiedMemoryHandler
from airbornehrs.core import AdaptiveFrameworkConfig, PerformanceSnapshot, FeedbackBuffer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OGD_Verify")

def run_verification():
    logger.info("🧪 Verifying Immortal Memory (OGD)...")
    
    # Define a simple model
    model = nn.Sequential(
        nn.Linear(10, 10, bias=False) # Simple linear model for easy subspace checking
    )
    
    # Initialize Memory with OGD
    memory = UnifiedMemoryHandler(
        model,
        method='hybrid',
        use_ogd=True
    )
    
    if not memory.projector:
        logger.error("❌ OGD Projector not initialized.")
        return

    # Create Dummy Data for Task A
    logger.info("Generating Task A Data...")
    X_a = torch.randn(100, 10)
    # Make X_a low rank to have a clear null space?
    # Or just random. 100 samples in 10D space -> Full rank usually.
    # But we set threshold=0.95, so it might pick top k components.
    
    # Create a mock feedback buffer
    config = AdaptiveFrameworkConfig()
    buffer = FeedbackBuffer(config, 'cpu')
    
    # Add samples to buffer
    for i in range(100):
        buffer.add(
            input_args=(X_a[i:i+1],),
            input_kwargs={},
            output=torch.randn(1, 10),
            target=torch.randn(1, 10),
            reward=0.0,
            loss=0.0
        )
        
    # Consolidate (Compute Subspace)
    logger.info("Consolidating Task A...")
    memory.consolidate(feedback_buffer=buffer)
    
    # Check if subspace exists
    layer_name = '0' # nn.Sequential names layers '0', '1'...
    if layer_name not in memory.projector.subspaces:
        # Try finding the name
        keys = list(memory.projector.subspaces.keys())
        logger.info(f"Subspaces found for: {keys}")
        if not keys:
            logger.error("❌ No subspaces computed.")
            return
        layer_name = keys[0]
        
    M = memory.projector.subspaces[layer_name]
    logger.info(f"Subspace shape for {layer_name}: {M.shape}")
    
    # Test Projection
    logger.info("Testing Gradient Projection...")
    # Random gradient
    grad = torch.randn(10, 10) # [Out, In]
    
    # Project
    grad_proj = memory.projector.project_gradient(layer_name, grad)
    
    # Verify Orthogonality
    # Rows of grad_proj should be orthogonal to M
    # (grad_proj * M) should be 0
    # grad_proj: [10, 10], M: [10, k]
    # product: [10, k]
    
    product = torch.mm(grad_proj, M)
    norm = product.norm().item()
    
    logger.info(f"Projection Residual Norm: {norm:.6f}")
    
    if norm < 1e-4:
        logger.info("✅ Gradient is orthogonal to Task A subspace.")
    else:
        logger.error(f"❌ Gradient projection failed. Residual: {norm}")
        return

    logger.info("✅ OGD Verification Passed!")

if __name__ == "__main__":
    run_verification()
