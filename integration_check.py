#!/usr/bin/env python3
"""Core framework integration check"""
import sys
sys.path.insert(0, '.')

from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

config = AdaptiveFrameworkConfig()
framework = AdaptiveFramework(model, config)

# Check integrations
checks = {
    'EWC (Memory)': hasattr(framework, 'memory') and hasattr(framework.memory, 'consolidate'),
    'SI Accumulation': hasattr(framework, 'memory') and hasattr(framework.memory, 'accumulate_path'),
    'Introspection': hasattr(framework, 'introspection_engine'),
    'Consciousness': hasattr(framework, 'consciousness'),
    'Consolidation Scheduler': hasattr(framework, 'consolidation_scheduler'),
    'Feedback Buffer': hasattr(framework, 'feedback_buffer'),
    'Meta Controller': hasattr(framework, 'meta_controller'),
    'Adapter Bank': hasattr(framework, 'adapter_bank'),
    'Telemetry Bus': hasattr(framework, 'telemetry_buffer'),
    'Monitor': hasattr(framework, 'monitor'),
}

print('\n=== FRAMEWORK INTEGRATION CHECK ===\n')
for name, status in checks.items():
    symbol = 'OK' if status else 'FAIL'
    print(f'[{symbol}] {name}')

all_good = all(checks.values())
print(f'\n{"="*50}')
print(f'{"OK: ALL SYSTEMS ONLINE" if all_good else "ERROR: MISSING COMPONENTS"}')
print(f'{"="*50}')
