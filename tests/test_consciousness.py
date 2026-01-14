import pytest
import torch
from airbornehrs.consciousness_v2 import EnhancedConsciousnessCore, MemoryEpisode

@pytest.fixture
def consciousness():
    """Fixture for consciousness core."""
    return EnhancedConsciousnessCore(
        feature_dim=10,
        num_heads=2,
        awareness_buffer_size=50,
        novelty_threshold=1.0
    )

def test_consciousness_observe_basic(consciousness):
    """Test that observe returns a valid metrics dictionary."""
    y_true = torch.randn(5, 1)
    y_pred = torch.randn(5, 1)
    features = torch.randn(5, 10)
    
    metrics = consciousness.observe(
        y_true=y_true,
        y_pred=y_pred,
        features=features,
        task_id="test_task"
    )
    
    assert isinstance(metrics, dict)
    assert 'surprise' in metrics
    assert 'uncertainty' in metrics
    assert 'emotion' in metrics
    assert 0.0 <= metrics['confidence'] <= 1.0

def test_system2_trigger(consciousness, mocker):
    """Test that Global Workspace (System 2) thinking is triggered on high uncertainty."""
    # Force high uncertainty via features (using mock or high entropy logic inside)
    # Actually, we can just spy on the global workspace forward pass
    
    spy = mocker.spy(consciousness.global_workspace, 'forward')
    
    # Create chaotic input to trigger uncertainty if possible, 
    # OR manually set internal state if needed. 
    # For now, let's just run observe and check if workspace is called at least once
    # defaults might trigger it if uncertainty is non-zero.
    
    y_true = torch.randn(5, 1)
    y_pred = torch.randn(5, 1) * 10 # High error
    features = torch.randn(5, 10)
    
    consciousness.observe(y_true=y_true, y_pred=y_pred, features=features)
    
    # Workspace might be called if confusion > 0
    # The code says if uncertainty > 0.5 or error_mean > 1.0
    # Our error will be high due to *10, so it should trigger.
    spy.assert_called()

def test_episodic_memory_storage(consciousness):
    """Verify episodes are stored and retrieved."""
    x = torch.randn(1, 10)
    consciousness.episodic_memory.store_episode(
        x=x,
        error=0.1,
        surprise=0.5,
        learning_gain=1.0,
        emotional_state="curious",
        task_difficulty=0.5
    )
    
    assert len(consciousness.episodic_memory.episodes) == 1
    
    # Retrieval
    memories = consciousness.episodic_memory.retrieve_relevant_memories(
        current_surprise=0.5,
        current_error=0.1,
        k=1
    )
    assert len(memories) == 1
    assert memories[0].emotional_state == "curious"
