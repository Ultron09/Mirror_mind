import pytest
from airbornehrs.core import AdaptiveFrameworkConfig
from airbornehrs.validation import validate_config

def test_config_validator_valid():
    config = AdaptiveFrameworkConfig()
    is_valid, _, _ = validate_config(config, raise_on_error=False)
    assert is_valid

def test_config_validator_invalid():
    config = AdaptiveFrameworkConfig(learning_rate=-0.01)
    
    # Should raise ValueError if raise_on_error=True (default)
    with pytest.raises(ValueError):
        validate_config(config, raise_on_error=True)
