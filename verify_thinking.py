import torch
import torch.nn as nn
from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig

# Mock Agent meant to be "Confused"
class ConfusedAgent(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.layer = nn.Linear(dim, dim)
        
    def forward(self, x):
        # Outputs roughly uniform logits -> High Entropy
        return torch.zeros_like(x) 

def verify_system_2():
    print("="*60)
    print("AIRBORNE HRS V9.3 - METACOGNITIVE INFERENCE CHECK")
    print("="*60)
    
    dim = 16
    model = ConfusedAgent(dim)
    config = AdaptiveFrameworkConfig.production()
    config.model_dim = dim
    config.num_heads = 4
    config.use_graph_memory = True
    config.enable_consciousness = True # Needed for Entropy
    
    framework = AdaptiveFramework(model, config, device='cpu')
    
    # 1. Inject some knowledge into memory first (so System 2 has something to recall)
    print("[SETUP] Injecting knowledge...")
    known_vec = torch.ones(1, dim) * 5.0 # Distinct knowledge
    
    # Manually add to memory so we store the "Correct" target (5.0), not the model's 0.0 prediction
    snapshot = type('Snapshot', (), {})()
    snapshot.input_args = (known_vec,) # Context
    snapshot.target = known_vec # The "Truth" we want to recall
    snapshot.timestamp = 0.0
    
    # We index it using the "Ambiguous" key so it matches the query
    ambiguous_key = torch.zeros(1, dim)
    
    if hasattr(framework.memory, 'graph_memory') and framework.memory.graph_memory:
        framework.memory.graph_memory.add(snapshot, ambiguous_key)
        print("   [INFO] Injected manual memory with target=5.0.")
    
    # 2. Run Cognitive Inference on an Ambiguous Input
    ambiguous_input = torch.zeros(1, dim) # Matches model's confusion
    
    print("\n[TEST] Running cognitive_inference (Threshold=0.0)...")
    # Threshold 0.0 forces System 2 activation (since entropy > 0)
    pred, diag = framework.cognitive_inference(ambiguous_input, threshold=-1.0)
    
    print(f"   Mode: {diag.get('mode')}")
    print(f"   Initial Uncertainty: {diag.get('initial_uncertainty', 0.0):.4f}")
    print(f"   Memories Retrieved: {diag.get('retrieved_memories')}")
    
    if diag.get('mode') == 'System 2 (Deliberative)':
        print("   [PASS] System 2 Activated.")
        if diag.get('retrieved_memories', 0) > 0:
             print("   [PASS] Active Recall Triggered.")
             # Check if output shifted towards memory (Consensus)
             # Prediction was Zero. Memory was 5.0.
             # Refined should be > 0.
             avg_val = pred.mean().item()
             print(f"   Refined Prediction Mean: {avg_val:.4f} (Original: 0.0)")
             if avg_val > 0.1:
                 print("   [PASS] Consensus applied (Prediction shifted towards memory).")
             else:
                 print("   [WARN] Consensus weak.")
        else:
             print("   [WARN] No memories found (context too distinct?).")
    else:
        print("   [FAIL] System 2 NOT Activated.")

if __name__ == "__main__":
    verify_system_2()
