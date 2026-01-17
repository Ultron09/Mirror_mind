
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

# Force local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig
from airbornehrs.production import ProductionAdapter, InferenceMode

def get_mini_data():
    # Larger dataset for streaming
    # Task A: 0..0.4 -> Label 0
    x_a = torch.rand(500, 10) * 0.4
    y_a = torch.zeros(500).long()
    
    # Task B: 0.6..1.0 -> Label 1
    x_b = torch.rand(500, 10) * 0.4 + 0.6
    y_b = torch.ones(500).long()
    
    return x_a, y_a, x_b, y_b

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 2)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

def run_mock_inference():
    torch.manual_seed(42)
    device = 'cpu'
    
    # Configure for Production
    cfg = AdaptiveFrameworkConfig(device=device, learning_rate=0.005)
    cfg.memory_type = 'hybrid'
    cfg.ewc_lambda = 2000.0
    cfg.si_lambda = 10.0
    cfg.use_prioritized_replay = True # Still needed for EWC buffer
    cfg.dream_batch_size = 32
    # Note: Dream interval doesn't matter if ProductionAdapter disables dreaming, 
    # BUT EWC regularization should still work.
    
    model = AdaptiveFramework(SimpleNet(), cfg, device=device)
    
    # 1. PRE-TRAIN TASK A (Offline)
    print("Pre-training Task A (Offline)...")
    x_a, y_a, x_b, y_b = get_mini_data()
    loader_a = DataLoader(TensorDataset(x_a, y_a), batch_size=32, shuffle=True)
    
    for epoch in range(5):
        for x, y in loader_a:
            model.train_step(x.to(device), target_data=y.to(device))
            
    # Consolidate Memory (Lock Task A)
    model.memory.consolidate(feedback_buffer=model.prioritized_buffer)
    print("Task A Consolidated.")
    
    # 2. DEPLOYMENT (Online Learning Task B)
    print("Deploying Model (InferenceMode.BUFFERED)...")
    adapter = ProductionAdapter(model, inference_mode=InferenceMode.BUFFERED)
    
    # Streaming Simulation
    # We feed Task B samples one by one (or small batches)
    # We check Retention (Task A) every N steps
    
    history = {
        'step': [],
        'acc_a': [],
        'acc_b_cum': [] # Cumulative accuracy on B
    }
    
    stream_size = 200
    b_correct = 0
    b_total = 0
    
    # Validation Set for A (Fixed)
    val_x_a = x_a[:100].to(device)
    val_y_a = y_a[:100].to(device)
    
    for i in range(stream_size):
        # Incoming Request (Task B)
        bx = x_b[i].unsqueeze(0).to(device) # Single sample batch
        by = y_b[i].unsqueeze(0).to(device)
        
        # Deploy: Predict + Update
        # Ideally production doesn't have labels immediately, but for "Online Learning" 
        # we assume feedback loop (Self-Supervised or Delayed Feedback).
        # Here we assume immediate feedback (Label provided).
        out = adapter.predict(bx, update=True, target=by)
        
        # Track simulated performance
        pred = out.argmax(dim=1)
        if pred.item() == by.item():
            b_correct += 1
        b_total += 1
        
        # Monitoring
        if (i+1) % 10 == 0:
            # Check Retention
            with torch.no_grad():
                out_a, _, _ = model(val_x_a)
                if isinstance(out_a, tuple): out_a = out_a[0]
                acc_a = (out_a.argmax(dim=1) == val_y_a).float().mean().item() * 100
            
            acc_b_rolling = (b_correct / b_total) * 100
            
            history['step'].append(i+1)
            history['acc_a'].append(acc_a)
            history['acc_b_cum'].append(acc_b_rolling)
            
            print(f"Step {i+1}: Task A Ret={acc_a:.1f}%, Task B Learn={acc_b_rolling:.1f}%")

    # 3. PLOT
    plt.figure(figsize=(10, 6))
    plt.plot(history['step'], history['acc_a'], label='Stability (Task A Retention)', color='#95a5a6', linewidth=3)
    plt.plot(history['step'], history['acc_b_cum'], label='Plasticity (Task B Learning)', color='#2ecc71', linewidth=3)
    
    plt.title("Mock Deployment: Online Learning vs Retention")
    plt.xlabel("Inference Steps (Stream)")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(os.path.dirname(__file__), 'mock_inference_results.png')
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")

if __name__ == "__main__":
    run_mock_inference()
