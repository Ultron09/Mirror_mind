
import torch
import torch.nn as nn
import torch.nn.functional as F
from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig

# 1. Define the Embeddings Model (Same as Benchmark)
class ManifoldModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

def debug_embeddings_loss():
    print("\n--- Debugging Embeddings Loss ---")
    input_dim = 2
    hidden_dim = 32
    output_dim = 2
    batch_size = 32
    
    model = ManifoldModel(input_dim, hidden_dim, output_dim)
    
    # Config matching the benchmark
    config = AdaptiveFrameworkConfig(
        model_dim=hidden_dim,
        use_prioritized_replay=True,
        use_gradient_centralization=True,
        learning_rate=1e-2
    )
    agent = AdaptiveFramework(model, config)
    
    # Generate random data (Gaussian Clusters)
    inputs = torch.randn(batch_size, input_dim)
    targets = torch.randn(batch_size, output_dim) # Random targets for debug
    
    print(f"Inputs shape: {inputs.shape}")
    print(f"Targets shape: {targets.shape}")
    
    # 1. Baseline Loss Check (Manual)
    model.eval()
    with torch.no_grad():
        out = model(inputs)
        base_loss = F.mse_loss(out, targets)
        print(f"Baseline Initial Loss (Manual Calc): {base_loss.item():.4f}")
        
    # 2. Agent Train Loop Analysis
    print("\n--- Running Agent Train Loop (50 steps) ---")
    model.train()
    
    # Initial step to get baseline metrics
    metrics = agent.train_step(inputs, target_data=targets)
    initial_loss = metrics['loss']
    print(f"Initial Loss: {initial_loss:.4f}")
    
    for step in range(50):
        # New random batch each time (simulating data stream)
        inputs = torch.randn(batch_size, input_dim)
        targets = torch.randn(batch_size, output_dim)
        
        metrics = agent.train_step(inputs, target_data=targets)
        
        if step % 10 == 0:
            print(f"Step {step}: Loss {metrics['loss']:.4f}")
            
    final_loss = metrics['loss']
    print(f"Final Loss: {final_loss:.4f}")
    
    if final_loss > initial_loss:
        print("❌ LOSS INCREASED! Divergence detected.")
    else:
        print("✅ Loss decreased or stable.")

if __name__ == "__main__":
    debug_embeddings_loss()
