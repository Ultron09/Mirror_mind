
import torch
import torch.nn as nn
import torch.nn.functional as F
from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig

# 1. Define the NLP Model (Same as Benchmark)
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

def debug_nlp_loss():
    print("\n--- Debugging NLP Loss Explosion ---")
    vocab_size = 50
    seq_len = 10
    batch_size = 4
    hidden_dim = 64
    
    model = SimpleSeqModel(vocab_size=vocab_size, hidden_dim=hidden_dim)
    
    # Config matching the benchmark but stripped down
    config = AdaptiveFrameworkConfig(
        model_dim=hidden_dim,
        enable_consciousness=False, # Disable
        use_lookahead=False,        # Disable
        use_gradient_centralization=False, # Disable
        memory_type='dnc',
        learning_rate=1e-2
    )
    agent = AdaptiveFramework(model, config)
    
    # Generate random data
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"Inputs shape: {inputs.shape}")
    print(f"Targets shape: {targets.shape}")
    
    # 1. Baseline Loss Check (Manual)
    model.eval()
    with torch.no_grad():
        out = model(inputs)
        base_loss = F.cross_entropy(out.reshape(-1, vocab_size), targets.reshape(-1))
        print(f"Baseline Initial Loss (Manual Calc): {base_loss.item():.4f}")
        
    # 2. Agent Train Loop Analysis
    print("\n--- Running Agent Train Loop (50 steps) ---")
    model.train()
    
    # Initial step
    metrics = agent.train_step(inputs, target_data=targets)
    initial_loss = metrics['loss']
    print(f"Initial Loss: {initial_loss:.4f}")
    
    for step in range(50):
        # New random batch (Copy Task: Target = Input)
        # For Copy Task, we need pattern repetition, not random noise every time if we want to test memorization of specific sequences?
        # Actually, Copy Task usually means: Input [A, B, C], Output [A, B, C]. The pattern is "Identity".
        # So random inputs are fine, as long as Target == Input.
        
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = inputs.clone() # COPY TASK
        
        metrics = agent.train_step(inputs, target_data=targets)
        
        if step % 10 == 0:
            print(f"Step {step}: Loss {metrics['loss']:.4f}")
            
    final_loss = metrics['loss']
    print(f"Final Loss: {final_loss:.4f}")
    
    if final_loss > 3.0:
        print("❌ LOSS STUCK HIGH! Not learning.")
    else:
        print("✅ Loss decreasing.")

if __name__ == "__main__":
    debug_nlp_loss()
