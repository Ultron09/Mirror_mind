
import sys
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import copy

# Force local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig

# --- SYNTHETIC DATA GENERATOR (Split-MNIST Style) ---
def get_synthetic_tasks(num_tasks=5, dim=100, samples_per_task=200):
    tasks = []
    # Generate 10 classes worth of "proto-types"
    prototypes = torch.randn(num_tasks * 2, dim) # 2 classes per task
    
    for t in range(num_tasks):
        # Task t has classes 2*t and 2*t+1
        c1, c2 = 2*t, 2*t+1
        
        # Generate samples near prototypes
        x1 = prototypes[c1] + torch.randn(samples_per_task // 2, dim) * 0.3
        y1 = torch.zeros(samples_per_task // 2).long() # Label 0 relative to task head
        
        x2 = prototypes[c2] + torch.randn(samples_per_task // 2, dim) * 0.3
        y2 = torch.ones(samples_per_task // 2).long()  # Label 1 relative to task head
        
        x = torch.cat([x1, x2])
        y = torch.cat([y1, y2])
        
        # Shuffle
        idx = torch.randperm(samples_per_task)
        tasks.append((x[idx], y[idx]))
    return tasks

class SimpleMLP(nn.Module):
    def __init__(self, in_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2) # Binary head for simplicity
        )
    def forward(self, x):
        return self.net(x)

def evaluate(model, task_data):
    x, y = task_data
    model.model.eval()
    with torch.no_grad():
        out, _, _ = model(x)
        pred = out.argmax(dim=1)
        acc = (pred == y).float().mean().item() * 100
    return acc

def run_experiment(mode='naive'):
    print(f"Running Experiment: {mode.upper()}")
    device = 'cpu'
    tasks = get_synthetic_tasks()
    
    cfg = AdaptiveFrameworkConfig(device=device)
    if mode == 'airborne':
        cfg.memory_type = 'hybrid'
        cfg.ewc_lambda = 2000.0
        cfg.dream_interval = 2
        cfg.dream_batch_size = 32
        cfg.enable_consciousness = True
    else:
        cfg.memory_type = 'none'
        cfg.enable_consciousness = False
        
    model = AdaptiveFramework(SimpleMLP(), cfg, device=device)
    
    # Track accuracy of ALL tasks after EACH task training
    history = [] 
    
    for t_idx, (x_train, y_train) in enumerate(tasks):
        print(f"  Training Task {t_idx+1}...")
        
        # Train
        batches = 5
        bs = 32
        for _ in range(batches):
             for i in range(0, len(x_train), bs):
                 bx, by = x_train[i:i+bs], y_train[i:i+bs]
                 model.train_step(bx, target_data=by)
                 
        # Consolidate (if airborne)
        if mode == 'airborne':
            model.memory.consolidate(feedback_buffer=model.prioritized_buffer)
            
        # Eval Phase: Check Avg Accuracy on Tasks 0 .. t_idx
        # But actually, standard is to check ALL tasks seen so far
        avg_acc = 0
        seen_tasks = t_idx + 1
        for old_t in range(seen_tasks):
            acc = evaluate(model, tasks[old_t])
            avg_acc += acc
        
        history.append(avg_acc / seen_tasks)
        print(f"  Task {t_idx+1} Done. Avg Acc (Seen): {history[-1]:.1f}%")
        
    return history

def plot_paper_figures():
    # 1. THE FORGETTING CURVE
    naive_hist = run_experiment('naive')
    airborne_hist = run_experiment('airborne')
    
    tasks = range(1, 6)
    
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(tasks, airborne_hist, 'o-', color='#2ecc71', linewidth=3, label='AirborneHRS (Ours)')
    ax.plot(tasks, naive_hist, 's--', color='#e74c3c', linewidth=2, label='Baseline (Naive)')
    
    ax.set_title("Retention Capacity: Sequential Task Learning", fontsize=14, pad=15)
    ax.set_xlabel("Number of Tasks Learned", fontsize=12)
    ax.set_ylabel("Average Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_xticks(tasks)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=12)
    
    # Branding
    ax.text(0.95, 0.05, "AirborneHRS v2.0", transform=ax.transAxes, ha='right', alpha=0.3)
    
    out_path = os.path.join(os.path.dirname(__file__), 'paper_fig1_forgetting.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Figure 1 saved to: {out_path}")

if __name__ == "__main__":
    plot_paper_figures()
