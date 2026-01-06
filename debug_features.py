
import torch
import torch.nn as nn
from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig

# 1. Define a simple sequence model for NLP reproduction
class SimpleSeqModel(nn.Module):
    def __init__(self, vocab_size=50, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.fc(out)

def test_nlp_sequence_handling():
    print("\n--- Testing NLP Sequence Handling ---")
    vocab_size = 50
    seq_len = 10
    batch_size = 4
    
    model = SimpleSeqModel(vocab_size=vocab_size)
    config = AdaptiveFrameworkConfig(
        model_dim=32,
        enable_consciousness=True,
        memory_type='dnc',
        use_lookahead=True
    )
    agent = AdaptiveFramework(model, config)
    
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len)) # Target is sequence
    
    print(f"Inputs shape: {inputs.shape}")
    print(f"Targets shape: {targets.shape}")
    
    try:
        # Pass targets directly as sequence
        metrics = agent.train_step(inputs, target_data=targets)
        print("✅ train_step completed successfully.")
        print(f"Loss: {metrics['loss']}")
        print(f"Z-Score: {metrics['z_score']}")
    except Exception as e:
        print(f"❌ train_step FAILED: {e}")
        import traceback
        traceback.print_exc()

def test_lookahead_activation():
    print("\n--- Testing Lookahead Activation ---")
    model = nn.Linear(10, 2)
    config = AdaptiveFrameworkConfig(use_lookahead=True)
    agent = AdaptiveFramework(model, config)
    
    # Check if slow weights are initialized
    if hasattr(agent, 'slow_weights') and len(agent.slow_weights) > 0:
        print("✅ Slow weights initialized.")
    else:
        print("❌ Slow weights NOT initialized.")

    # Run steps to trigger lookahead
    inputs = torch.randn(4, 10)
    targets = torch.randn(4, 2)
    
    initial_slow_weight = agent.slow_weights['weight'].clone()
    
    # Lookahead k=5, so run 6 steps to see update
    for i in range(6):
        agent.train_step(inputs, target_data=targets)
        
    final_slow_weight = agent.slow_weights['weight']
    
    if not torch.equal(initial_slow_weight, final_slow_weight):
        print("✅ Lookahead update CONFIRMED (slow weights changed).")
    else:
        print("❌ Lookahead update FAILED (slow weights did not change).")

if __name__ == "__main__":
    test_nlp_sequence_handling()
    test_lookahead_activation()
