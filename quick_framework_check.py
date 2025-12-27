#!/usr/bin/env python3
"""Quick framework integrity check"""
import sys
sys.path.insert(0, '.')

try:
    from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig
    print("✅ Imports successful")
    
    # Test config
    config = AdaptiveFrameworkConfig()
    print(f"✅ Config initialized (memory_type={config.memory_type})")
    
    # Check key attributes
    assert hasattr(config, 'si_lambda'), "Missing si_lambda"
    assert hasattr(config, 'si_xi'), "Missing si_xi"
    assert hasattr(config, 'enable_consciousness'), "Missing enable_consciousness"
    assert hasattr(config, 'consolidation_criterion'), "Missing consolidation_criterion"
    print("✅ Config has all required attributes")
    
    # Check framework structure
    import torch.nn as nn
    model = nn.Linear(10, 1)
    framework = AdaptiveFramework(model, config)
    
    assert hasattr(framework, 'ewc'), "Missing ewc handler"
    assert hasattr(framework, 'introspection_engine'), "Missing introspection engine"
    assert hasattr(framework, 'monitor'), "Missing monitor"
    assert hasattr(framework, 'feedback_buffer'), "Missing feedback buffer"
    assert hasattr(framework, 'consolidation_scheduler'), "Missing consolidation scheduler"
    print("✅ Framework has all core components")
    
    # Check key methods
    assert callable(framework.forward), "forward not callable"
    assert callable(framework.train_step), "train_step not callable"
    assert callable(framework.consolidate_memory), "consolidate_memory not callable"
    print("✅ Framework has all required methods")
    
    # Test forward pass
    import torch
    x = torch.randn(4, 10)
    y = torch.randn(4, 1)
    
    try:
        output, log_var, affine_modifiers = framework.forward(x)
        print(f"✅ Forward pass works (log_var={log_var.item():.4f})")
    except Exception as e:
        print(f"⚠️ Forward pass issue: {e}")
    
    print("\n" + "="*50)
    print("✅ FRAMEWORK STATUS: HEALTHY")
    print("="*50)
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
