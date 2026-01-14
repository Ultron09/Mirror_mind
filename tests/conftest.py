import pytest
import torch
import torch.nn as nn
from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig
from airbornehrs.presets import PRESETS

@pytest.fixture
def simple_model():
    """A lightweight model for fast logic testing."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )

@pytest.fixture
def framework_config():
    """Fast config for CPU-based testing."""
    preset = PRESETS.fast()
    config_dict = preset.to_dict()
    
    # Overrides
    config_dict['device'] = "cpu"
    config_dict['enable_consciousness'] = True
    config_dict['enable_world_model'] = True
    config_dict['feedback_buffer_size'] = 50
    config_dict['model_dim'] = 10
    config_dict['num_heads'] = 2
    
    return AdaptiveFrameworkConfig(**config_dict)

@pytest.fixture
def framework(simple_model, framework_config):
    """Instantiated framework with simple model."""
    return AdaptiveFramework(
        user_model=simple_model,
        config=framework_config,
        device="cpu"
    )
