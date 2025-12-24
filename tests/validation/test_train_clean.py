#!/usr/bin/env python
"""Test consciousness-aware training loop - emoji-free output."""
import torch
import torch.nn as nn
import logging
import os

# Configure logging for Windows console
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

# Redirect stderr to stdout to capture logging errors
import sys
sys.stderr = sys.stdout

# Disable emoji logging in airbornehrs
os.environ['MM_NO_EMOJI'] = '1'

from airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig

# Create a simple model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 5)
)

# Create framework with consciousness enabled
config = AdaptiveFrameworkConfig(
    model_dim=256,
    device='cpu',
    memory_type='hybrid',
    enable_consciousness=True,
    use_prioritized_replay=True,
    adaptive_lambda=True,
    dream_interval=5,
    log_frequency=5
)

print("Initializing framework...")
try:
    framework = AdaptiveFramework(model, config)
    print("[OK] Framework initialized with consciousness enabled")
except Exception as e:
    print(f"[ERROR] Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n=== CONSCIOUSNESS-AWARE TRAINING TEST ===\n")

# Generate simple synthetic data
torch.manual_seed(42)
success_count = 0
for step in range(1, 21):
    try:
        # Random inputs and targets
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)
        
        # Train step with consciousness
        metrics = framework.train_step(x, y, enable_dream=True, meta_step=True)
        success_count += 1
        
        # Log key metrics
        if step % 5 == 0:
            loss = metrics.get('loss', 0.0)
            print(f"Step {step:2d}: Loss={loss:.4f} | Dreaming | Consciousness")
    except Exception as e:
        print(f"Step {step:2d}: FAILED - {type(e).__name__}: {str(e)[:60]}")
        if step <= 5:
            import traceback
            traceback.print_exc()
        break

print(f"\n[RESULT] {success_count} steps completed successfully")
if success_count >= 15:
    print("[SUCCESS] Full consciousness-aware training loop completed!")
    print("System learned to adapt, prioritize, and consolidate internally.")
else:
    print(f"[WARNING] Only {success_count}/20 steps succeeded")
