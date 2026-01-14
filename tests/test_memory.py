import pytest
import torch
import torch.nn as nn
from airbornehrs.memory import UnifiedMemoryHandler, PrioritizedReplayBuffer, RelationalGraphMemory
from airbornehrs.core import PerformanceSnapshot

@pytest.fixture
def mock_model():
    return nn.Sequential(nn.Linear(10, 10))

def test_unified_memory_consolidate(mock_model):
    """Check that consolidation runs without crashing."""
    handler = UnifiedMemoryHandler(mock_model, method='si', feature_dim=10)
    
    # Create a fake buffer
    # snapshot = PerformanceSnapshot(...) 
    # Actually, EWC needs a buffer. Let's test the 'si' path which doesn't STRICTLY need buffer for consolidate()
    # But let's test basic init and existence first.
    
    assert handler.is_enabled() == False
    
    # Manually inject some importance to simulate "learning"
    for n, p in mock_model.named_parameters():
        handler.omega[n] = torch.ones_like(p)
        
    assert handler.is_enabled() == True
    
    # Consolidation step (SI logic)
    handler.consolidate(current_step=1)
    assert handler.last_consolidation_step == 1

def test_prioritized_replay():
    """Test that high priority items are sampled more often (probabilistically)."""
    buffer = PrioritizedReplayBuffer(capacity=100, temperature=1.0)
    
    # Mock snapshots
    class MockSnapshot:
        def __init__(self, imp): self.importance = imp; self.z_score = 0; self.age_in_steps = 0
    
    # Add one high importance, one low
    high = MockSnapshot(100.0)
    low = MockSnapshot(0.01)
    
    buffer.add(high, importance=100.0)
    buffer.add(low, importance=0.01)
    
    # Sample many times
    high_count = 0
    for _ in range(100):
        batch = buffer.sample_batch(1)
        if batch[0].importance == 100.0:
            high_count += 1
            
    # High importance item should be sampled significantly more
    assert high_count > 70 

def test_graph_memory_linking():
    """Test that nodes form links based on similarity."""
    graph = RelationalGraphMemory(feature_dim=4, link_threshold=0.5)
    
    # Add Node A
    feat_a = torch.tensor([1.0, 0.0, 0.0, 0.0])
    graph.add(snapshot="A", feature_vector=feat_a)
    
    # Add Node B (Orthogonal, should not link)
    feat_b = torch.tensor([0.0, 0.0, 0.0, 1.0])
    graph.add(snapshot="B", feature_vector=feat_b)
    
    # Add Node C (Similar to A, should link)
    feat_c = torch.tensor([0.9, 0.1, 0.0, 0.0])
    graph.add(snapshot="C", feature_vector=feat_c)
    
    # Check links
    # Node C (index 2) should link to A (index 0)
    # We might need to check the internal structure or use retrieve to verify association
    
    # Retrieve using A's feature
    results = graph.retrieve(feat_a, k=5)
    
    # Needs to return A, and potentially C via association
    # snapshot strings "A", "C"
    assert "A" in results
    # Ideally "C" should be there too if linked
    # We can check internal links for strict logic testing
    node_c = graph.nodes[2]
    # Check if any link points to 0 (Node A)
    linked_indices = [l[0] for l in node_c.links]
    assert 0 in linked_indices
