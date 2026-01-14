import pytest
import torch
import torch.nn as nn
import threading
from airbornehrs.production import ProductionAdapter, InferenceMode
from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig

@pytest.fixture
def framework_prod():
    """Framework for production tests."""
    from airbornehrs.presets import PRESETS
    preset = PRESETS.fast()
    config_dict = preset.to_dict()
    config_dict['device'] = 'cpu'
    config_dict['model_dim'] = 10
    config_dict['num_heads'] = 2
    
    config = AdaptiveFrameworkConfig(**config_dict)
    
    model = nn.Linear(10, 1)
    return AdaptiveFramework(model, config, device='cpu')

def test_manual_train_step(framework_prod):
    """Sanity check that train_step works before threading."""
    inputs = torch.randn(5, 10)
    targets = torch.randn(5, 1)
    metrics = framework_prod.train_step(inputs, target_data=targets)
    assert 'loss' in metrics

def test_production_predict_static(framework_prod):
    """Test static prediction (no update)."""
    adapter = ProductionAdapter(framework_prod, inference_mode=InferenceMode.STATIC)
    
    inputs = torch.randn(5, 10)
    output = adapter.predict(inputs)
    
    assert output is not None
    assert output.shape == (5, 1)

def test_thread_safety_buffered(framework_prod):
    """Test concurrent requests to buffered adapter."""
    adapter = ProductionAdapter(framework_prod, inference_mode=InferenceMode.BUFFERED)
    adapter.buffer_size = 10
    
    def worker():
        inputs = torch.randn(1, 10)
        targets = torch.randn(1, 1)
        adapter.predict(inputs, update=True, target=targets)
        
    threads = []
    for _ in range(20):
        t = threading.Thread(target=worker)
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    # Check that buffer was flushed (size should be 0 or small remnant)
    # 20 items, buffer 10 -> should have flushed twice.
    # Remainder 0.
    assert len(adapter.inference_buffer) == 0
