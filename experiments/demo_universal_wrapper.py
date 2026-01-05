import torch
import torch.nn as nn
import sys
import os
import logging

# Path Setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import airbornehrs
print(f"DEBUG: airbornehrs path: {airbornehrs.__file__}")

from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig

print(f"DEBUG: Config Annotations: {AdaptiveFrameworkConfig.__annotations__.keys()}")
print(f"DEBUG: Config Init Signature: {AdaptiveFrameworkConfig.__init__}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UniversalWrapperDemo")

# 1. Define a "Dumb" Standard Model
class DumbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.classifier = nn.Linear(32, 2)
        
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def run_demo():
    logger.info("🤖 Defining a standard 'Dumb' PyTorch Model...")
    dumb_model = DumbModel()
    
    # Verify it's just a normal model
    logger.info(f"Original Type: {type(dumb_model)}")
    
    # 2. The Magic: Wrap it
    logger.info("\n✨ WRAPPING WITH AIRBORNE.HRS (Protocol V7)...")
    
    config = AdaptiveFrameworkConfig(
        model_dim=32, # Match internal feature dim for best results
        use_moe=True, # Inject Mixture of Experts?
        enable_consciousness=True, # Give it a soul?
        device='cpu'
    )
    
    # ONE LINE TO RULE THEM ALL
    smart_model = AdaptiveFramework(dumb_model, config)
    
    # 3. Verify Superpowers
    logger.info(f"\n🚀 Transformed Type: {type(smart_model)}")
    
    # Check for V7 Features
    if hasattr(smart_model, 'consciousness') and smart_model.consciousness:
        logger.info("✅ Consciousness: ONLINE (Global Workspace Active)")
        
    if hasattr(smart_model, 'memory'):
        logger.info("✅ Immortal Memory: ONLINE (OGD/EWC Ready)")
        
    if hasattr(smart_model, 'meta_controller'):
        logger.info("✅ Metacognition: ONLINE (Learned Optimizer Ready)")
        
    # 4. Run a Training Step
    logger.info("\n🏃 Running Training Step...")
    x = torch.randn(5, 10)
    y = torch.tensor([0, 1, 0, 1, 0])
    
    # The wrapper handles the complex training loop
    metrics = smart_model.train_step(x, target_data=y)
    
    logger.info(f"Loss: {metrics['loss']:.4f}")
    logger.info(f"Emotion: {metrics['mode']}")
    logger.info(f"Surprise: {metrics['z_score']:.4f}")
    
    logger.info("\n🎉 SUCCESS: The 'Dumb' model is now a SOTA Agent.")

if __name__ == "__main__":
    run_demo()
