import torch
import torch.nn as nn
import time
import threading
from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16)
    def forward(self, x):
        return self.fc(x)

def verify_dreaming():
    print("="*60)
    print("AIRBORNE HRS V9.4 - PROACTIVE DREAMING CHECK")
    print("="*60)
    
    model = SimpleModel()
    config = AdaptiveFrameworkConfig.production()
    config.model_dim = 16
    config.num_heads = 4
    config.enable_dreaming = True
    
    framework = AdaptiveFramework(model, config, device='cpu')
    
    # 1. Populate Buffer (Simulate Inference)
    print("[SETUP] Populating experience buffer...")
    for _ in range(20):
        x = torch.randn(1, 16)
        y = x # Identity task
        # Manually add to buffer
        snapshot = framework.feedback_buffer.add(
            input_args=(x,),
            input_kwargs={},
            output=y,
            target=y,
            reward=1.0, 
            loss=0.0
        )
        
        # Also populate prioritized buffer if it exists (Framework prefers it)
        if framework.prioritized_buffer:
             # Need to reconstruct snapshot or valid object
             # feedback_buffer.add might not return the snapshot depending on implementation
             # but we can create one
             snap = type('Snapshot', (), {})()
             snap.input_args = (x,)
             snap.target = y
             snap.output = y
             snap.loss = 0.0
             snap.reward = 1.0
             snap.timestamp = 0.0
             framework.prioritized_buffer.add(snap, z_score=1.0) # High priority error

        
    # 2. Capture Weight State
    initial_weights = model.fc.weight.clone()
    
    # 3. Trigger Dream (Proactive Step)
    print("[TEST] Triggering 'Dream' cycle...")
    framework.learn_from_buffer(batch_size=5, num_epochs=1)
    
    # 4. Check Impact
    final_weights = model.fc.weight
    delta = (final_weights - initial_weights).abs().sum().item()
    
    print(f"   Weight Delta: {delta:.6f}")
    
    if delta > 0:
        print("   [PASS] Model updated weights proactively using stored memories.")
    else:
        print("   [FAIL] Weights stayed static (Dreaming failed).")

if __name__ == "__main__":
    verify_dreaming()
