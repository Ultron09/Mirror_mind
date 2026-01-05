
import torch
import torch.nn as nn
from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig
import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Debug')

def test_repro():
    logger.info("🧪 Starting Repro...")
    
    # Simple Model
    model = nn.Linear(16, 2)
    framework = AdaptiveFramework(model, AdaptiveFrameworkConfig(model_dim=16, enable_consciousness=True))
    
    x = torch.randn(4, 16).to(framework.device)
    y = torch.randint(0, 2, (4,)).to(framework.device) # Long targets
    
    logger.info(f"Input shape: {x.shape}, Target shape: {y.shape}, Target dtype: {y.dtype}")
    
    try:
        metrics = framework.train_step(x, target_data=y)
        logger.info("✅ Train Step Passed")
        
        # Add to buffer manually to ensure consolidation has data
        # output must be tensor
        dummy_out = torch.randn(4, 2).to(framework.device)
        framework.feedback_buffer.add(x, {}, dummy_out, y, 0.0, metrics['loss'])
        
        logger.info("🧪 Testing Consolidation...")
        framework.consolidate_memory(mode='NORMAL')
        logger.info("✅ Consolidation Passed")
        
    except Exception as e:
        logger.error(f"❌ Train Step Failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_repro()
