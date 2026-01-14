import pytest
import torch
import torch.nn as nn
from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig
from pathlib import Path
import os

def test_initialization(simple_model):
    """Test that the framework initializes components correctly."""
    config = AdaptiveFrameworkConfig(
        model_dim=10,
        num_heads=2,
        enable_consciousness=True,
        memory_type='hybrid',
        device='cpu'
    )
    framework = AdaptiveFramework(simple_model, config, device='cpu')
    
    assert framework.model is not None
    assert framework.consciousness is not None
    assert framework.memory is not None
    assert framework.introspection_engine is not None

def test_train_step_output(framework):
    """Test that train_step returns the expected dictionary."""
    inputs = torch.randn(5, 10) # [Batch, Dim]
    targets = torch.randn(5, 1)
    
    metrics = framework.train_step(inputs, target_data=targets)
    
    assert isinstance(metrics, dict)
    assert 'loss' in metrics
    assert 'z_score' in metrics
    assert isinstance(metrics['loss'], float)

def test_plasticity_mechanics(framework):
    """Test that affine modifiers are generated and have correct shape."""
    inputs = torch.randn(1, 10)
    
    # Run forward pass manually to check returns
    _, log_var, affine_modifiers = framework.forward(inputs)
    
    # Affine modifiers should be [2] (scale, shift) or [Batch, 2]
    assert affine_modifiers is not None
    assert affine_modifiers.shape[-1] == 4 # Core.py IntrospectionEngine outputs 4 dims now, or acts on [GlobalState] -> [Mu, Sigma]

def test_save_and_load_checkpoint(framework, tmp_path):
    """Test full checkpointing cycle."""
    save_path = tmp_path / "test_checkpoint.pt"
    
    # Train once to change state
    inputs = torch.randn(5, 10)
    targets = torch.randn(5, 1)
    framework.train_step(inputs, target_data=targets)
    
    # Save
    framework.save_checkpoint(str(save_path))
    assert save_path.exists()
    
    # Load into new instance
    new_model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    new_framework = AdaptiveFramework.load_checkpoint(
        str(save_path),
        model=new_model,
        device='cpu'
    )
    
    assert new_framework.step_count == framework.step_count
    # Check config preservation
    assert new_framework.config.model_dim == framework.config.model_dim

def test_optimizer_mocking(mocker, framework):
    """Ensure optimizer.step() is actually called."""
    # Mock the optimizer
    mock_step = mocker.patch.object(framework.optimizer, 'step')
    
    inputs = torch.randn(5, 10)
    targets = torch.randn(5, 1)
    
    framework.train_step(inputs, target_data=targets)
    
    mock_step.assert_called_once()
