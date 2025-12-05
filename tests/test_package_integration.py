import torch
import pytest
import sys
import os

# The test assumes the package 'airbornehrs' is installed in the environment.
# This prevents testing the local folder by accident, ensuring we test the artifact.

def test_package_structure():
    """
    Verifies that the package exposes the correct public API classes.
    This ensures that __init__.py is correctly configured.
    """
    try:
        import airbornehrs
    except ImportError:
        pytest.fail("Could not import airbornehrs package. Is it installed via 'pip install -e.'?")
    
    # Verify core components are exposed
    assert hasattr(airbornehrs, 'AdaptiveFramework'), "Package missing AdaptiveFramework export"
    assert hasattr(airbornehrs, 'MetaController'), "Package missing MetaController export"
    assert hasattr(airbornehrs, 'ProductionAdapter'), "Package missing ProductionAdapter export"
    
    # Verify that EasyTrainer is correctly EXCLUDED (it's legacy)
    assert not hasattr(airbornehrs, 'EasyTrainer'), "Legacy EasyTrainer should not be in the package"

def test_training_cycle_smoke():
    """
    Smoke test: Constructs Framework and MetaController, 
    and runs a single training/adaptation step on synthetic data.
    This validates the 'Optimization Cycle' logic.
    """
    from airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig, MetaController

    # 1. Setup Minimal Config
    config = AdaptiveFrameworkConfig(
        model_dim=64,  # Small dimension for speed
        num_layers=2,
        num_heads=2,
        batch_size=4,
        learning_rate=0.01
    )

    # 2. Instantiate (using CPU for CI compatibility)
    device = 'cpu'
    framework = AdaptiveFramework(config, device=device)
    
    # Attach the Meta-Controller (The Stabilizer)
    meta_controller = MetaController(framework)

    # 3. Create Dummy Data
    # Shape: (Batch, Sequence Length, Feature Dim)
    X = torch.randn(8, 10, 64) 
    y = torch.randn(8, 10, 64)

    # 4. Execution Phase: Training Step
    # The train_step method returns a dict of metrics
    try:
        metrics = framework.train_step(X, y)
    except Exception as e:
        pytest.fail(f"AdaptiveFramework.train_step failed: {e}")

    # Validation of Training Output
    assert isinstance(metrics, dict), "train_step should return a dict"
    assert 'loss' in metrics, "metrics should contain 'loss'"
    assert metrics['loss'] > 0, "Loss should be a positive float"

    # 5. Execution Phase: Adaptation Step
    # This tests the GradientAnalyzer and LR Scheduler logic
    try:
        adaptation = meta_controller.adapt(loss=metrics['loss'])
    except Exception as e:
        pytest.fail(f"MetaController.adapt failed: {e}")
        
    # Validation of Adaptation Output
    assert isinstance(adaptation, dict), "Adaptation step should return metrics"
    assert 'learning_rate' in adaptation, "Meta-controller should report new LR"
    
    # 6. Verify Self-Improvement Mechanism
    # The adaptation step should return a learning rate (even if unchanged)
    assert isinstance(adaptation['learning_rate'], float)