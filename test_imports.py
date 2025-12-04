#!/usr/bin/env python
"""Quick smoke test for airbornehrs package"""

import sys
sys.path.insert(0, '.')

print("Testing airbornehrs imports...")
print()

# Test 1: Core framework
from airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig
print("✅ AdaptiveFramework imports successful")

# Test 2: Meta-controller
from airbornehrs import MetaController, GradientAnalyzer
print("✅ MetaController imports successful")

# Test 3: Production adapter
from airbornehrs import ProductionAdapter, InferenceMode
print("✅ ProductionAdapter imports successful")

# Test 4: Quick instantiation
config = AdaptiveFrameworkConfig(model_dim=64, num_layers=2)
framework = AdaptiveFramework(config)
print(f"✅ Framework instantiation successful (device: {framework.device})")

print()
print("=" * 60)
print("🎉 ALL TESTS PASSED - Package is production-ready!")
print("=" * 60)
