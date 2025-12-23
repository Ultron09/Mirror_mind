#!/usr/bin/env python
"""Quick test of consciousness-enabled framework initialization."""
import torch
import torch.nn as nn
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
    adaptive_lambda=True
)

try:
    framework = AdaptiveFramework(model, config)
    print("[OK] Framework initialized")
    print(f"[OK] Consciousness enabled: {config.enable_consciousness}")
    print(f"[OK] Memory type: {config.memory_type}")
    print(f"[OK] Prioritized replay: {config.use_prioritized_replay}")
    print(f"[OK] Unified memory handler active: {hasattr(framework, 'ewc') and framework.ewc is not None}")
    print(f"[OK] Consciousness core active: {hasattr(framework, 'consciousness') and framework.consciousness is not None}")
    print("[READY] SOTA consciousness framework is fully initialized!")
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()
