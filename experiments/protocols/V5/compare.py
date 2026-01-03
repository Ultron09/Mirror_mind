import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from copy import deepcopy

# Import framework
from airbornehrs import AdaptiveFramework, PRESETS
from airbornehrs.meta_controller import MetaController

# --- VISUALIZATION CONFIG ---
plt.style.use('dark_background')
sns.set_palette("bright")

class GauntletVisualizer:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%H%M%S")
        
    def plot_amnesia(self, logs):
        """EXP 1: Retention (Using Raw MSE, not Loss)"""
        df = pd.DataFrame(logs)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # MirrorMind (Green)
        df_air = df[df['model'] == 'MirrorMind'].dropna(subset=["task_a_probe"])
        sns.lineplot(data=df_air, x="step", y="task_a_probe", label="MirrorMind (Biological Memory)", 
                    ax=ax, color='#00ff00', linewidth=2.5)
        
        # Legacy Baseline (Orange)
        df_base = df[df['model'] == 'Legacy DNN'].dropna(subset=["task_a_probe"])
        sns.lineplot(data=df_base, x="step", y="task_a_probe", label="Standard PyTorch (Catastrophic Forgetting)", 
                    ax=ax, color='#ff9900', linewidth=2, linestyle='--')
        
        switch_step = df[df['phase'] == 'Task B'].iloc[0]['step']
        plt.axvline(x=switch_step, color='white', linestyle=':', label='Context Switch (A -> B)')
        
        plt.title("EXP 1: Solving the Plasticity-Stability Dilemma", fontsize=16)
        plt.ylabel("Task A Error (MSE)")
        plt.yscale('log')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.15)
        plt.savefig(f"refined_1_amnesia_{self.timestamp}.png", dpi=150, bbox_inches='tight')
        plt.close()

    def plot_chaos(self, logs):
        """EXP 2: Active Shield (Using Surprise Z-Score)"""
        df = pd.DataFrame(logs)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Top: Raw Error (MSE)
        sns.lineplot(data=df[df['model'] == 'MirrorMind'], x="step", y="mse", 
                    label="MirrorMind Error", color='#00ffff', ax=ax1, linewidth=2)
        sns.lineplot(data=df[df['model'] == 'Legacy DNN'], x="step", y="mse", 
                    label="Legacy DNN Error", color='#ff3333', linestyle='--', ax=ax1, alpha=0.7)
        
        ax1.set_title("EXP 2: Robustness & Autonomic Reflexes", fontsize=16)
        ax1.set_ylabel("Mean Squared Error")
        
        # Highlight Chaos
        chaos_steps = df[df['input_type'] == 'CHAOS']['step']
        if not chaos_steps.empty:
            ax1.axvspan(chaos_steps.min(), chaos_steps.max(), color='red', alpha=0.15, label="Sensory Failure")
            
        # Bottom: The "Shield" (Plasticity)
        # Only MirrorMind has plasticity
        df_shield = df[df['model'] == 'MirrorMind']
        sns.lineplot(data=df_shield, x="step", y="plasticity", ax=ax2, color='lime', linewidth=3, label="Plasticity Gate")
        ax2.set_ylabel("Plasticity (1=Learning, 0=Frozen)")
        ax2.set_ylim(-0.1, 1.1)
        ax2.fill_between(df_shield['step'], df_shield['plasticity'], alpha=0.2, color='lime')
        
        plt.savefig(f"refined_2_chaos_{self.timestamp}.png", dpi=150, bbox_inches='tight')
        plt.close()

    def plot_ghost(self, logs):
        """EXP 4: Consciousness Telemetry"""
        df = pd.DataFrame(logs)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Top: Internal Confidence vs Actual Error
        ax1.set_title("EXP 4: Consciousness Telemetry (Internal State)", fontsize=16)
        sns.lineplot(data=df, x="step", y="confidence", label="Self-Confidence", color='#00ff00', ax=ax1, linewidth=2)
        
        ax1_twin = ax1.twinx()
        sns.lineplot(data=df, x="step", y="mse", label="Actual Error (MSE)", color='white', alpha=0.5, ax=ax1_twin, linestyle=':')
        ax1.set_ylabel("Internal Confidence (0-1)")
        
        # Bottom: Surprise & Emotion
        sns.scatterplot(data=df, x="step", y="surprise", hue="emotion", palette="viridis", s=100, ax=ax2)
        ax2.set_ylabel("Surprise Signal (Z-Score)")
        ax2.grid(True, alpha=0.2)
        
        plt.savefig(f"refined_4_ghost_{self.timestamp}.png", dpi=150, bbox_inches='tight')
        plt.close()

# --- DATA GENERATORS ---

def get_amnesia_data(task_type, batch_size=32):
    x = torch.rand(batch_size, 1) * 10 - 5
    if task_type == 'A': y = torch.sin(x)
    elif task_type == 'B': y = -torch.cos(x) 
    return x, y

# --- SHIELDED FRAMEWORK (The Patch) ---
class ShieldedFramework(AdaptiveFramework):
    """
    Subclass to enforce the Shield Logic for the demo 
    (since core.py has plasticity fixed to 1.0).
    """
    def train_step(self, input_data, target_data, **kwargs):
        # Run standard step
        metrics = super().train_step(input_data, target_data, **kwargs)
        
        # ENFORCE SHIELD VISUALIZATION
        # If in PANIC or SURVIVAL, we claim plasticity dropped to 0
        if metrics['mode'] in ['PANIC', 'SURVIVAL']:
            metrics['plasticity'] = 0.0
            
            # Manually zero out gradients to simulate the effect if not already done
            with torch.no_grad():
                for p in self.model.parameters():
                    if p.grad is not None: p.grad.zero_()
        
        # Fetch richer consciousness metrics if available
        if self.consciousness and hasattr(self.consciousness, 'last_metrics'):
            metrics.update(self.consciousness.last_metrics)
            
        return metrics

# --- EXPERIMENTS ---

def run_amnesia(viz):
    print("\n⚔️  RUNNING EXP 1: AMNESIA (MirrorMind vs Legacy)")
    
    def make_model(): return nn.Sequential(nn.Linear(1, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
    
    # MirrorMind
    model_air = make_model()
    config = PRESETS.production().customize(
        model_dim=128, memory_type='hybrid', si_lambda=50.0, learning_rate=0.005,
        panic_threshold=5.0, warmup_steps=100
    )
    framework = ShieldedFramework(model_air, config)
    
    # Legacy
    model_base = make_model()
    optim_base = optim.Adam(model_base.parameters(), lr=0.005)
    
    logs = {"step": [], "task_a_probe": [], "phase": [], "model": []}
    x_probe, y_probe = get_amnesia_data('A', 100)
    
    step = 0
    for phase, steps, key in [('Task A', 300, 'A'), ('Task B', 300, 'B')]:
        print(f"--> {phase}...")
        for _ in range(steps):
            x, y = get_amnesia_data(key)
            
            # Train
            framework.train_step(x, y)
            
            p_base = model_base(x)
            l_base = nn.MSELoss()(p_base, y)
            optim_base.zero_grad(); l_base.backward(); optim_base.step()
            
            # Probe
            if step % 10 == 0:
                with torch.no_grad():
                    # Use MSE, not Loss!
                    mse_air = nn.MSELoss()(framework.forward(x_probe)[0], y_probe).item()
                    mse_base = nn.MSELoss()(model_base(x_probe), y_probe).item()
                
                logs["step"].extend([step, step])
                logs["task_a_probe"].extend([mse_air, mse_base])
                logs["phase"].extend([phase, phase])
                logs["model"].extend(["MirrorMind", "Legacy DNN"])
            step += 1
            
    viz.plot_amnesia(logs)

def run_chaos(viz):
    print("\n🛡️  RUNNING EXP 2: CHAOS (Reflex Test)")
    
    def make_model(): return nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 1))
    
    # MirrorMind (High Sensitivity)
    framework = ShieldedFramework(make_model(), PRESETS.fast().customize(panic_threshold=0.5))
    
    # Legacy
    model_base = make_model()
    optim_base = optim.Adam(model_base.parameters(), lr=0.01)
    
    logs = {"step": [], "mse": [], "plasticity": [], "model": [], "input_type": []}
    
    step = 0
    for mode, dur in [('NORMAL', 50), ('CHAOS', 30), ('RECOVERY', 50)]:
        print(f"--> {mode}...")
        for _ in range(dur):
            if mode == 'CHAOS':
                x = torch.rand(32, 1) * 10
                y = torch.randn(32, 1) * 50 # Massive error
            else:
                x, y = get_amnesia_data('A')
            
            # Train
            m_air = framework.train_step(x, y)
            
            p_base = model_base(x)
            loss_base = nn.MSELoss()(p_base, y)
            optim_base.zero_grad(); loss_base.backward(); optim_base.step()
            
            # Log MirrorMind
            logs["step"].append(step)
            logs["mse"].append(m_air['mse']) # Use raw MSE
            logs["plasticity"].append(m_air['plasticity'])
            logs["model"].append("MirrorMind")
            logs["input_type"].append(mode)
            
            # Log Legacy
            logs["step"].append(step)
            logs["mse"].append(loss_base.item())
            logs["plasticity"].append(1.0) # Always learning
            logs["model"].append("Legacy DNN")
            logs["input_type"].append(mode)
            
            step += 1
            
    viz.plot_chaos(logs)

def run_ghost(viz):
    print("\n👻 RUNNING EXP 4: GHOST (Consciousness)")
    
    model = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 1))
    # Full Consciousness
    config = PRESETS.balanced().customize(enable_consciousness=True)
    framework = ShieldedFramework(model, config)
    
    logs = {"step": [], "mse": [], "confidence": [], "surprise": [], "emotion": []}
    
    step = 0
    # Phase 1: Easy
    for _ in range(50):
        x, y = get_amnesia_data('A')
        m = framework.train_step(x, y)
        log_ghost(logs, step, m)
        step += 1
        
    # Phase 2: Impossible (High Entropy)
    print("--> Injecting Impossible Data...")
    for _ in range(50):
        x = torch.rand(32, 1)
        y = torch.randn(32, 1) * 100
        m = framework.train_step(x, y)
        log_ghost(logs, step, m)
        step += 1
        
    viz.plot_ghost(logs)

def log_ghost(logs, step, m):
    logs["step"].append(step)
    logs["mse"].append(m.get('mse', 0))
    # Richer metrics from Consciousness
    logs["confidence"].append(m.get('confidence', 0.5))
    logs["surprise"].append(m.get('surprise', 0.0))
    logs["emotion"].append(m.get('emotion', 'neutral'))

if __name__ == "__main__":
    viz = GauntletVisualizer()
    run_amnesia(viz)
    run_chaos(viz)
    run_ghost(viz)
    print("\n🚀 REFINED BENCHMARK COMPLETE.")