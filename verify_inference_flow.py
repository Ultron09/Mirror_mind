import torch
import torch.nn as nn
import os
import shutil
from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig

# 1. Define Protocol (Simple Regression: y = x * 2)
class CognitiveAgent(nn.Module):
    def __init__(self, input_dim=10, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

def run_lifecycle_test():
    print("="*60)
    print("AIRBORNE HRS V9.1 - LIFECYCLE VERIFICATION (Train -> Deploy)")
    print("="*60)
    
    # Setup
    ckpt_path = "checkpoints/test_deploy.pt"
    if os.path.exists("checkpoints"):
        shutil.rmtree("checkpoints")
    
    # --- PHASE 1: TRAINING ---
    print("\n[PHASE 1] Training 'Sentient' Agent on Pattern (y = 2x)...")
    
    # Use divisible dimensions for MultiheadAttention
    base_model = CognitiveAgent(16, 16)
    config = AdaptiveFrameworkConfig.production()
    config.model_dim = 16
    config.num_heads = 4 # 16 / 4 = 4
    config.use_moe = True
    config.enable_consciousness = True
    
    framework = AdaptiveFramework(base_model, config, device='cpu')
    
    for i in range(100):
        # Generate Data: y = 2x
        x = torch.randn(4, 16)
        y = x * 2
        
        metrics = framework.train_step(x, target_data=y, enable_dream=False)
        
        if i % 20 == 0:
            print(f"   Step {i}: Loss = {metrics['loss']:.4f}, Emotion = {metrics.get('dominant_emotion', 'N/A')}")
            
    print("   [INFO] Training Complete.")
    
    # --- PHASE 2: DEPLOYMENT (Save & Load) ---
    print("\n[PHASE 2] Saving Brain to Checkpoint...")
    framework.save_checkpoint(ckpt_path)
    del framework # Simulate server shutdown
    print(f"   [INFO] Brain saved to {ckpt_path}")
    
    print("\n[PHASE 3] Simulating Production Server Startup...")
    # New separate instance
    # Must match config
    new_base_model = CognitiveAgent(16, 16)
    deployed_node = AdaptiveFramework(new_base_model, config, device='cpu')
    deployed_node.load_checkpoint(ckpt_path)
    print("   [INFO] Brain Reloaded Successfully.")
    
    # --- PHASE 4: INFERENCE TESTING ---
    print("\n[PHASE 4] Running Inference Test Cases...")
    
    test_cases = [
        torch.ones(1, 16),       # Expect ~2.0
        torch.ones(1, 16) * -1,  # Expect ~-2.0
        torch.randn(1, 16)       # Random
    ]

    
    for idx, case in enumerate(test_cases):
        print(f"\n   Case {idx+1}: Input Mean = {case.mean().item():.2f}")
        
        # USE THE NEW INFERENCE STEP
        pred, diagnostics = deployed_node.inference_step(case, return_diagnostics=True)
        
        expected_mean = case.mean().item() * 2
        actual_mean = pred.mean().item()
        error = abs(actual_mean - expected_mean)
        
        print(f"      -> Prediction Mean: {actual_mean:.4f} (Expected: {expected_mean:.4f})")
        print(f"      -> Error: {error:.4f}")
        
        # Check Diagnostics
        cons = diagnostics.get('consciousness', {})
        print(f"      -> Consciousness: {cons.get('dominant_emotion', 'Unknown')} (Alertness: {cons.get('alertness', 0):.2f})")
        
        if 'expert_usage' in diagnostics:
            usage = diagnostics['expert_usage']
            print(f"      -> Experts Activated: {usage.sum()} hits")

    print("\n[VERDICT] System verified? Yes/No")
    if error < 0.5:
        print("   YES - System successfully learned, saved, loaded, and served predictions.")
    else:
        print("   NO - High error rate detected.")

if __name__ == "__main__":
    run_lifecycle_test()
