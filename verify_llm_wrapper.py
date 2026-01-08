import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig

# 1. Mock a HuggingFace-style LLM
@dataclass
class MockCausalLMOutput:
    logits: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor]] = None

class MiniGPT(nn.Module):
    """Simulates a tiny Language Model outputting (Batch, Seq, Vocab)"""
    def __init__(self, vocab_size=100, embed_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.Linear(embed_dim, embed_dim) # Dummy transformer
        self.head = nn.Linear(embed_dim, vocab_size)
        self.embed_dim = embed_dim
        
    def forward(self, input_ids):
        # input_ids: [Batch, Seq]
        x = self.embedding(input_ids)
        feat = self.transformer(x) # [Batch, Seq, Dim]
        logits = self.head(feat)   # [Batch, Seq, Vocab]
        
        # Return object like HF
        return MockCausalLMOutput(logits=logits, hidden_states=(feat,))

def test_llm_integration():
    print("="*60)
    print("AIRBORNE HRS V9.1 - LLM COMPATIBILITY CHECK")
    print("="*60)
    
    # 1. Setup
    vocab_size = 100
    model_dim = 16
    llm = MiniGPT(vocab_size, model_dim)
    
    config = AdaptiveFrameworkConfig.production()
    config.model_dim = model_dim
    config.num_heads = 4
    config.enable_consciousness = True # We want to verify this works on LLMs
    config.use_graph_memory = True     # Verify memory works with tokens
    
    # 2. Add 'Expert' capabilities to the LLM
    config.use_moe = False # Keep it simple for now, or True to transform
    
    wrapper = AdaptiveFramework(llm, config, device='cpu')
    print("[INFO] LLM Wrapped successfully.")
    
    # 3. Inference Test (No Training)
    print("\n[TEST] Running Inference Step (Zero-Shot)...")
    
    # Fake Token IDs [Batch=1, Seq=5]
    input_ids = torch.randint(0, vocab_size, (1, 5))
    
    # Run Inference
    # The framework should detect 'logits' and 'hidden_states' automatically
    logits, diagnostics = wrapper.inference_step(input_ids, return_diagnostics=True)
    
    print(f"   Input Shape: {input_ids.shape}")
    print(f"   Logits Shape: {logits.shape}")
    
    # 4. Check for 'Consciousness'
    cons = diagnostics.get('consciousness', {})
    print(f"   [DIAGNOSTIC] Consciousness State Present? {'Yes' if cons else 'No'}")
    if cons:
        print(f"   -> Dominant Emotion: {cons.get('dominant_emotion')}")
        print(f"   -> Entropy (Confusion): {cons.get('entropy', 0):.4f}")
        
    # 5. Check Memory
    if 'expert_usage' in diagnostics:
        print("   [DIAGNOSTIC] Expert Usage Tracked.")
        
    print("\n[VERDICT]")
    if logits.shape == (1, 5, vocab_size) and cons:
        print("PASS - Framework successfully processed LLM structure and generated cognitive diagnostics.")
    else:
        print("FAIL - Did not handle LLM outputs correctly.")

if __name__ == "__main__":
    test_llm_integration()
