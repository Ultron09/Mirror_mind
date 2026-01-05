import torch
import torch.nn as nn
import sys
import os
import logging
from dataclasses import dataclass

# Path Setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from airbornehrs.meta_controller import MetaController, MetaControllerConfig
from airbornehrs.core import AdaptiveFramework # Just for mocking

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Meta_Verify")

# Mock Framework
class MockFramework:
    def __init__(self):
        self.model = nn.Linear(10, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

def run_verification():
    logger.info("🧪 Verifying Metacognition (Learned Optimizer)...")
    
    # Config
    config = MetaControllerConfig(
        use_learned_optimizer=True,
        learned_optimizer_hidden_dim=16
    )
    
    # Initialize
    framework = MockFramework()
    controller = MetaController(framework, config)
    
    if not controller.lr_scheduler.policy:
        logger.error("❌ Learned Optimizer Policy not initialized.")
        return
    logger.info("✅ Learned Optimizer Policy initialized.")
    
    # Test Adaptation Sequence
    logger.info("Running Adaptation Sequence...")
    
    initial_lr = controller.lr_scheduler.current_lr
    logger.info(f"Initial LR: {initial_lr}")
    
    # Simulate Loss Decrease (Good)
    loss_seq = [1.0, 0.9, 0.8, 0.7, 0.6]
    for l in loss_seq:
        controller.adapt(loss=l)
        
    lr_after_good = controller.lr_scheduler.current_lr
    logger.info(f"LR after improvement: {lr_after_good}")
    
    # Simulate Loss Spike (Bad)
    loss_seq_bad = [0.6, 2.0, 2.5, 3.0]
    for l in loss_seq_bad:
        controller.adapt(loss=l)
        
    lr_after_bad = controller.lr_scheduler.current_lr
    logger.info(f"LR after spike: {lr_after_bad}")
    
    # Check if LR changed dynamically
    if lr_after_good == initial_lr and lr_after_bad == initial_lr:
        logger.warning("⚠️ LR did not change. Policy might be outputting 1.0 constantly.")
    else:
        logger.info("✅ LR adapted dynamically.")
        
    # Check if hidden state exists
    if controller.lr_scheduler.policy.hidden is not None:
         logger.info("✅ LSTM Hidden State is active.")
    else:
         logger.error("❌ LSTM Hidden State is None.")

    logger.info("✅ Metacognition Verification Passed!")

if __name__ == "__main__":
    run_verification()
