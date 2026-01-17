
import sys
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Force local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig
print(f"DEBUG: Using Local Package: {os.path.dirname(os.path.abspath(__file__))}")
import airbornehrs
print(f"DEBUG: airbornehrs Loaded From: {os.path.dirname(airbornehrs.__file__)}")

# --- SIMULATION DATA ---
class EmployeeDatabase:
    def __init__(self, dim=64):
        self.dim = dim
        self.day_shift = []   # Classes 0-9
        self.night_shift = [] # Classes 10-19
        
    def generate_data(self):
        # HARD CORE MODE: Same distribution, shuffled pixels.
        # This guarantees Catastrophic Forgetting in Naive models
        # because the input statistics are identical (Mean 0, Var 1).
        
        # Base Prototypes (Day Shift)
        self.prototypes = torch.randn(10, self.dim) # Classes 0-9
        
        # Day Shift: Normal
        loader_day = self._make_loader(self.prototypes, 0, permute=False)
        
        # Night Shift: SAME Prototypes but Permuted (Pixel Shuffle)
        # This simulates "Same hardware, different encryption/pattern"
        loader_night = self._make_loader(self.prototypes, 10, permute=True)
        
        return loader_day, loader_night
        
    def _make_loader(self, prototypes, start_label, permute=False):
        xs, ys = [], []
        samples_per_person = 50
        
        perm_idx = torch.randperm(self.dim) if permute else torch.arange(self.dim)
        
        for i, p in enumerate(prototypes):
            # Variations
            x = p + torch.randn(samples_per_person, self.dim) * 0.2
            # Apply Permutation (Global for the shift)
            x = x[:, perm_idx]
            
            y = torch.full((samples_per_person,), start_label + i).long()
            xs.append(x)
            ys.append(y)
            
        x = torch.cat(xs)
        y = torch.cat(ys)
        
        idx = torch.randperm(len(x))
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x[idx], y[idx]), 
            batch_size=32, shuffle=True
        )

class SentinelNet(nn.Module):
    def __init__(self, input_dim=64, num_classes=20):
        super().__init__()
        # BRAIN UPGRADE: Efficient layers (Constraint) to force competition
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), # Medium capacity
            nn.ReLU(),
            nn.Linear(128, 64), # Bottleneck
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def run_simulation():
    print(">>> INITIALIZING PROJECT SENTINEL v2.0 (High Capacity)...")
    db = EmployeeDatabase()
    day_loader, night_loader = db.generate_data()
    
    # SETUP TWO SYSTEMS
    # Adjusted LR to prevent Adrenaline Overshoot
    systems = {
        'Legacy AI (Standard)': AdaptiveFrameworkConfig(
            memory_type='none', 
            device='cpu',
            learning_rate=0.001 # Slower, more stable
        ),
        'Sentinel AI (Airborne)': AdaptiveFrameworkConfig(
            memory_type='hybrid', 
            ewc_lambda=5000.0, 
            dream_interval=5, # Less aggressive dreaming (was 1)
            dream_batch_size=32,
            enable_consciousness=True,
            device='cpu',
            learning_rate=0.001
        )
    }
    
    audit_logs = {'Legacy AI (Standard)': [], 'Sentinel AI (Airborne)': []}
    
    for name, cfg in systems.items():
        print(f"\n[{name}] DEPLOYMENT STARTED.")
        model = AdaptiveFramework(SentinelNet(), cfg, device='cpu')
        
        # PHASE 1: DAY SHIFT ORIENTATION
        print(f"[{name}] Learning 'Day Shift' Personnel...")
        for _ in range(15): # More epochs for convergence
             for x, y in day_loader:
                 model.train_step(x, target_data=y)
        
        if 'Airborne' in name:
            model.memory.consolidate(feedback_buffer=model.prioritized_buffer)
            
        # AUDIT 1: CHECK DAY SHIFT
        acc_day_1 = evaluate(model, day_loader)
        print(f"[{name}] Audit 1 (Mid-Day): Day Shift Access = {acc_day_1:.1f}%")
        
        # PHASE 2: NIGHT SHIFT ORIENTATION (CRISIS)
        print(f"[{name}] Learning 'Night Shift' Personnel...")
        for _ in range(15):
             for x, y in night_loader:
                 model.train_step(x, target_data=y)
                 
        # AUDIT 2: CHECK BOTH SHIFTS
        acc_day_2 = evaluate(model, day_loader) # Did we lock them out?
        acc_night_2 = evaluate(model, night_loader) # Do we know the new guys?
        
        print(f"[{name}] Audit 2 (Midnight):")
        print(f"    Day Shift Access:   {acc_day_2:.1f}% (Retention)")
        print(f"    Night Shift Access: {acc_night_2:.1f}% (Plasticity)")
        
        audit_logs[name] = [acc_day_1, acc_day_2, acc_night_2]

    # Save numeric results for AI inspection
    res_path = os.path.join(os.path.dirname(__file__), 'audit_results.txt')
    with open(res_path, 'w') as f:
        f.write("SECURITY AUDIT RESULTS\n")
        f.write("======================\n")
        for name, vals in audit_logs.items():
             f.write(f"\nSYSTEM: {name}\n")
             f.write(f"  Audit 1 (Day): {vals[0]:.2f}%\n")
             f.write(f"  Audit 2 (Day-Retained): {vals[1]:.2f}%\n")
             f.write(f"  Audit 2 (Night-New):    {vals[2]:.2f}%\n")
    print(f"Results text saved to: {res_path}")

    # --- GENERATE REPORT (VISUALIZATION) ---
    print("\n>>> Generating Security Audit Report...")
    labels = ['Day Crew\n(Morning)', 'Day Crew\n(Midnight)', 'Night Crew\n(Midnight)']
    x = np.arange(len(labels))
    width = 0.35
    
    # Use Dark Theme for 'Coolness'
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, audit_logs['Legacy AI (Standard)'], width, label='Legacy System', color='#e74c3c')
    rects2 = ax.bar(x + width/2, audit_logs['Sentinel AI (Airborne)'], width, label='Sentinel AI', color='#2ecc71')
    
    ax.set_ylabel('Access Granted (%)', color='white')
    ax.set_title('Security Audit: Impact of Shift Changes (Hard Mode)', color='white')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 110)
    
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'security_audit.png')
    plt.savefig(save_path)
    print(f"Audit Report saved to: {save_path}")

def evaluate(model, loader):
    model.model.eval()
    c, t = 0, 0
    with torch.no_grad():
        for x, y in loader:
            out, _, _ = model(x)
            pred = out.argmax(dim=1)
            c += (pred == y).sum().item()
            t += y.size(0)
    return 100 * c / t

if __name__ == "__main__":
    run_simulation()
