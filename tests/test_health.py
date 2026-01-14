import pytest
import torch
import torch.nn as nn
from airbornehrs.health_monitor import NeuralHealthMonitor

def test_dead_neuron_detection():
    """Verify that zero-gradient layers are flagged as DEAD."""
    model = nn.Linear(10, 10)
    monitor = NeuralHealthMonitor(model, dead_threshold=1e-5)
    
    # Simulate forward/backward but zero out grads manually
    x = torch.randn(5, 10)
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    # Force grads to zero
    for p in model.parameters():
        if p.grad is not None:
             p.grad.zero_()
             
    report = monitor.check_vital_signs()
    
    # Should detect DEAD
    statuses = list(report.values())
    assert "DEAD" in statuses

def test_autonomic_repair():
    """Verify that repair resets parameters."""
    model = nn.Linear(10, 10)
    monitor = NeuralHealthMonitor(model)
    
    # Save old weights
    old_weight = model.weight.clone()
    
    # Create a report claiming DEAD
    report = {"weight": "DEAD"} # The key is parameter name, but monitor logic splits by '.'
    # Wait, the monitor expects full param names e.g. "weight" if it's top level, or "layer.weight"
    # For a simple Linear, it is "weight" and "bias"
    
    # Manually trigger repair
    # Note: autonomic_repair iterates names. For "weight", module_path is empty list?
    # Let's verify monitor logic: `module_path = name.split('.')[:-1]`
    # If name is "weight", path is []. target_mod = self.model (Linear). Correct.
    
    repairs = monitor.autonomic_repair({"weight": "DEAD"})
    
    assert repairs == 1
    assert not torch.equal(model.weight, old_weight) # Should be reset
