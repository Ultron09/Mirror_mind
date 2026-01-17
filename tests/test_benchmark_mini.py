
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest

# CRITICAL: Force use of local package, not pip installed one
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig

def get_mini_data():
    # Synthetic data: Task A (Inputs 0..0.5) -> Label 0
    #                 Task B (Inputs 0.5..1) -> Label 1
    # This ensures perfect task separation for testing forgetting
    
    # Task A
    x_a = torch.rand(100, 10) * 0.5 
    y_a = torch.zeros(100).long()
    
    # Task B
    x_b = torch.rand(100, 10) * 0.5 + 0.5
    y_b = torch.ones(100).long()
    
    return {
        'A': DataLoader(TensorDataset(x_a, y_a), batch_size=10, shuffle=True),
        'B': DataLoader(TensorDataset(x_b, y_b), batch_size=10, shuffle=True)
    }

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 2)
        
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

def run_mini_benchmark(mode='base'):
    torch.manual_seed(42)
    device = 'cpu' # Fast enough for mini test
    
    cfg = AdaptiveFrameworkConfig(device=device, learning_rate=0.01)
    if mode == 'base':
        cfg.memory_type = 'none'
        cfg.enable_consciousness = False
    else:
        # AIRBORNE CONFIG (The "Fix")
        cfg.memory_type = 'hybrid'
        cfg.enable_consciousness = True # Needed for SI/Plasticity
        cfg.use_prioritized_replay = True
        cfg.ewc_lambda = 2000.0 # Balanced Stability (was 5000)
        cfg.si_lambda = 10.0    # Moderate SI (was 100)
        cfg.dream_interval = 2  # Frequent Replay
        cfg.dream_batch_size = 10
        cfg.feedback_buffer_size = 100
        
    model = AdaptiveFramework(SimpleNet(), cfg, device=device)
    data = get_mini_data()
    
    # TASK A TRAINING
    print(f"\n[{mode.upper()}] Training Task A...")
    for epoch in range(5):
        for x, y in data['A']:
            model.train_step(x.to(device), target_data=y.to(device))
    
    # Consolidate (Crucial for SI/EWC)
    if mode == 'airborne':
        model.memory.consolidate(feedback_buffer=model.prioritized_buffer)
        
    # Eval Task A (Should be 100%)
    acc_a_p1 = evaluate(model, data['A'], device)
    print(f"[{mode.upper()}] Task A (Phase 1) Acc: {acc_a_p1:.1f}%")
    
    # TASK B TRAINING
    print(f"[{mode.upper()}] Training Task B...")
    for epoch in range(5):
        for x, y in data['B']:
            # In airborne, this will trigger dreaming every 2 steps
            model.train_step(x.to(device), target_data=y.to(device))
            
    # FINAL EVAL
    acc_a_p2 = evaluate(model, data['A'], device)
    acc_b_p2 = evaluate(model, data['B'], device)
    
    print(f"[{mode.upper()}] FINAL RESULTS:")
    print(f"  Task A Retention: {acc_a_p2:.1f}% (Delta: {acc_a_p2 - acc_a_p1:.1f})")
    print(f"  Task B Learning:  {acc_b_p2:.1f}%")
    
    return acc_a_p2, acc_b_p2

def evaluate(model, loader, device):
    model.model.eval()
    c, t = 0, 0
    with torch.no_grad():
        for x, y in loader:
            out, _, _ = model(x.to(device))
            if isinstance(out, tuple): out = out[0]
            pred = out.argmax(dim=1)
            c += (pred == y.to(device)).sum().item()
            t += y.size(0)
    return 100 * c / t

if __name__ == "__main__":
    print("Running Mini-Benchmark...")
    
    print(">>> MODE: BASE (Naive)")
    base_a, base_b = run_mini_benchmark('base')
    
    print("\n>>> MODE: AIRBORNE (Pro)")
    pro_a, pro_b = run_mini_benchmark('airborne')
    
    print("\n" + "="*30)
    print("VERDICT:")
    if pro_a > base_a + 20:
        print("PASS: Airborne successfully prevented forgetting!")
    else:
        print("FAIL: No significant improvement in retention.")
