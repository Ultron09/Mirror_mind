
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

def get_rich_data():
    # Task A: Sine Wave (Regression)
    t = torch.linspace(0, 10, 200).unsqueeze(1)
    y_a = torch.sin(t)
    
    # Task B: Square Wave (Regression) - Represents a "Shock"
    y_b = torch.sign(torch.sin(t)) * 1.5 # Larger amplitude
    
    return DataLoader(TensorDataset(t, y_a), batch_size=16, shuffle=True), \
           DataLoader(TensorDataset(t, y_b), batch_size=16, shuffle=True)

class RegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

def run_neuro_simulation():
    torch.manual_seed(42)
    device = 'cpu'
    
    cfg = AdaptiveFrameworkConfig(device=device, learning_rate=0.01)
    cfg.memory_type = 'hybrid'
    cfg.ewc_lambda = 2000.0
    cfg.si_lambda = 10.0
    cfg.enable_consciousness = True
    cfg.dream_interval = 2
    cfg.dream_batch_size = 16
    
    model = AdaptiveFramework(RegNet(), cfg, device=device)
    loader_a, loader_b = get_rich_data()
    
    history = {
        'step': [],
        'loss': [],
        'surprise': [],
        'plasticity': [],
        'phase': []
    }
    
    step = 0
    
    print(">>> PHASE 1: Living in a Sine Wave (Task A)...")
    for epoch in range(5):
        for x, y in loader_a:
            m = model.train_step(x, target_data=y)
            step += 1
            history['step'].append(step)
            history['loss'].append(m['loss'])
            history['surprise'].append(m.get('surprise', 0.0))
            history['plasticity'].append(m.get('plasticity', 1.0))
            history['phase'].append(0) # 0 = Task A
            
    print(">>> CONSOLIDATION (Sleeping)...")
    model.memory.consolidate(feedback_buffer=model.prioritized_buffer)
    
    print(">>> PHASE 2: The Shock (Task B - Square Wave)...")
    for epoch in range(5):
        for x, y in loader_b:
            m = model.train_step(x, target_data=y)
            step += 1
            history['step'].append(step)
            history['loss'].append(m['loss'])
            history['surprise'].append(m.get('surprise', 0.0))
            history['plasticity'].append(m.get('plasticity', 1.0))
            history['phase'].append(1) # 1 = Task B

    # --- PLOTTING ---
    print(">>> Generating Neuro-Dynamics Report...")
    
    plt.style.use('dark_background')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # X-Axis: Boundary
    boundary_step = [s for s, p in zip(history['step'], history['phase']) if p == 0][-1]
    
    # 1. LOSS (The External View)
    ax1.plot(history['step'], history['loss'], color='#e74c3c', linewidth=2, label='Error (Loss)')
    ax1.axvline(x=boundary_step, color='white', linestyle='--', alpha=0.5, label='Task Switch')
    ax1.set_title("External Performance: The Error Spike", fontsize=14, color='white')
    ax1.set_ylabel("loss")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.2)
    
    # 2. SURPRISE (The Perception)
    ax2.plot(history['step'], history['surprise'], color='#f1c40f', linewidth=2, label='Surprise (Entropy/Z-Score)')
    ax2.axvline(x=boundary_step, color='white', linestyle='--', alpha=0.5)
    ax2.fill_between(history['step'], history['surprise'], color='#f1c40f', alpha=0.2)
    ax2.set_title("Cognitive State: The Moment of Surprise", fontsize=14, color='white')
    ax2.set_ylabel("sigma (z)")
    ax2.grid(True, alpha=0.2)
    
    # 3. PLASTICITY (The Reaction)
    ax3.plot(history['step'], history['plasticity'], color='#00ffbd', linewidth=2, label='Adrenaline (LR Multiplier)')
    ax3.axvline(x=boundary_step, color='white', linestyle='--', alpha=0.5)
    ax3.fill_between(history['step'], history['plasticity'], color='#00ffbd', alpha=0.2)
    ax3.set_title("Biological Response: Adrenaline Injection", fontsize=14, color='white')
    ax3.set_ylabel("multiplier")
    ax3.set_xlabel("Experience Steps")
    ax3.grid(True, alpha=0.2)
    
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'neuro_dynamics_report.png')
    plt.savefig(save_path, dpi=150)
    print(f"Report saved to: {save_path}")

if __name__ == "__main__":
    run_neuro_simulation()
