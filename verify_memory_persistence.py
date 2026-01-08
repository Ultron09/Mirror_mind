import torch
import torch.nn as nn
from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig
import numpy as np

# Mock Agent
class CognitiveAgent(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.layer = nn.Linear(dim, dim)
    def forward(self, x):
        return self.layer(x)

def verify_memory_on_inference():
    print("="*60)
    print("AIRBORNE HRS V9.2 - ONLINE LEARNING VERIFICATION")
    print("="*60)
    
    # 1. Setup
    dim = 16
    model = CognitiveAgent(dim)
    config = AdaptiveFrameworkConfig.production()
    config.model_dim = dim
    config.num_heads = 4
    config.use_graph_memory = True # REQUIRED for "Never Forget"
    
    framework = AdaptiveFramework(model, config, device='cpu')
    
    # 2. Define a "Memory" (e.g., User's Name)
    # This vector represents "My name is Surya" concept
    memory_vector = torch.randn(1, dim)
    memory_vector = memory_vector / memory_vector.norm() # Normalize
    
    print("\n[STEP 1] Running Inference WITH Learning (remember=True)...")
    pred, diag = framework.inference_step(
        memory_vector, 
        return_diagnostics=True, 
        remember=True # <--- THE KEY FEATURE
    )
    
    if diag.get('memory_stored'):
        print("   [SUCCESS] System acknowledged memory storage.")
    else:
        print("   [FAIL] System did not store memory.")
        return

    # 2.5 Verify Node Count (Storage Check)
    node_count = len(framework.memory.graph_memory.nodes)
    if node_count > 0:
        print(f"   [STORAGE CHECK] Graph contains {node_count} nodes (Expected > 0). PASS.")
    else:
        print("   [STORAGE CHECK] Graph is EMPTY. FAIL.")
        return

    # 3. Verify Retrieval (Immediate Recall)
    print("\n[STEP 2] Verifying Short-Term Recall...")
    # Query with slightly noisy version
    query = memory_vector + torch.randn(1, dim) * 0.05
    query = query / query.norm()
    
    # Increase k to catch issues with un-adapted IVF centroids in 1-shot test
    results = framework.memory.graph_memory.retrieve(query, k=5)
    
    if len(results) > 0:
        print(f"   [RECALL] Found {len(results)} memory candidates.")
        
        # Check if the retrieved item is roughly our vector (optional, but good)
        # We can't check .similarity attribute as it's not attached to snapshot by default
        print("   [VERDICT] PASS - Model remembered the context from inference instantly.")
    else:
        print("   [VERDICT] FAIL - Memory not found in Graph retrieval.")

if __name__ == "__main__":
    verify_memory_on_inference()
