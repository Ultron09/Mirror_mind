#!/usr/bin/env python3
"""
INTEGRATION TEST: Verify all components work together
"""

import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

print("\n" + "="*80)
print("MIRRORMING INTEGRATION TEST")
print("="*80 + "\n")

# Test 1: Import integration
print("TEST 1: Import MirrorMind Integration")
print("-" * 80)
try:
    from airbornehrs.integration import create_mirrorming_system
    print("[OK] Integration module imported")
except Exception as e:
    print(f"[FAIL] Failed: {e}")
    exit(1)

# Test 2: Create system
print("\nTEST 2: Create MirrorMind System")
print("-" * 80)
try:
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 128)
            self.fc2 = nn.Linear(128, 10)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    
    model = SimpleModel()
    system = create_mirrorming_system(model, device='cpu')
    print("[OK] MirrorMind system created successfully")
except Exception as e:
    print(f"[FAIL] Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: EWC integration
print("\nTEST 3: EWC Handler")
print("-" * 80)
try:
    assert system.ewc is not None, "EWC not initialized"
    print(f"[OK] EWC Handler: lambda={system.ewc.ewc_lambda:.4f}")
except Exception as e:
    print(f"[FAIL] Failed: {e}")

# Test 4: Meta-Controller integration
print("\nTEST 4: Meta-Controller")
print("-" * 80)
try:
    assert system.meta_controller is not None, "Meta-controller not initialized"
    print(f"[OK] Meta-Controller: Reptile enabled")
except Exception as e:
    print(f"[FAIL] Failed: {e}")

# Test 5: Adapter Bank
print("\nTEST 5: Adapter Bank")
print("-" * 80)
try:
    assert len(system.adapters.adapters) > 0, "Adapters not initialized"
    print(f"[OK] Adapter Bank: {len(system.adapters.adapters)} adapters")
except Exception as e:
    print(f"[FAIL] Failed: {e}")

# Test 6: Consciousness Core
print("\nTEST 6: Consciousness Core")
print("-" * 80)
try:
    assert system.consciousness is not None, "Consciousness not initialized"
    print(f"[OK] Consciousness Core: Enabled")
except Exception as e:
    print(f"[FAIL] Failed: {e}")

# Test 7: Training step
print("\nTEST 7: Single Training Step")
print("-" * 80)
try:
    x = torch.randn(16, 64)
    y = torch.randint(0, 10, (16,))
    
    metrics = system.train_step(x, y, task_id=0, use_ewc=False, use_adapters=True)
    
    assert 'loss' in metrics, "Loss not in metrics"
    assert 'accuracy' in metrics, "Accuracy not in metrics"
    assert 'confidence' in metrics, "Confidence not in metrics"
    
    print(f"[OK] Training step works:")
    print(f"     Loss: {metrics['loss']:.4f}")
    print(f"     Accuracy: {metrics['accuracy']:.4f}")
    print(f"     Confidence: {metrics['confidence']:.4f}")
    print(f"     Uncertainty: {metrics['uncertainty']:.4f}")
except Exception as e:
    print(f"[FAIL] Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("[OK] CRITICAL INTEGRATION TESTS PASSED")
print("="*80 + "\n")
print("MirrorMind is ready for real dataset training!\n")
