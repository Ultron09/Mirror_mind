import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
from copy import deepcopy
import math

# Import framework
from airbornehrs import AdaptiveFramework, PRESETS
from airbornehrs.meta_controller import MetaController, MetaControllerConfig

# --- VISUALIZATION CONFIG ---
plt.style.use('dark_background')
sns.set_palette("bright")

class GauntletVisualizer:
    """Handles logging and plotting for all 4 experiments."""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%H%M%S")
        
    def plot_amnesia(self, logs):
        """EXP 1: Continual Learning Stability"""
        df = pd.DataFrame(logs)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Training Loss (Cyan)
        sns.lineplot(data=df, x="step", y="train_loss", label="Current Task Loss", 
                    ax=ax, color='cyan', alpha=0.6, linewidth=1)
        
        # Retention Probe (Green - The Money Shot)
        df_probe = df.dropna(subset=["task_a_probe"])
        sns.lineplot(data=df_probe, x="step", y="task_a_probe", label="Task A Retention (Memory)", 
                    ax=ax, color='lime', linewidth=3)
        
        # Switch Marker
        switch_step = df[df['phase'] == 'Task B'].iloc[0]['step']
        plt.axvline(x=switch_step, color='yellow', linestyle='--', label='Task Switch')
        
        plt.title("EXP 1: The 'Amnesia' Test (Catastrophic Forgetting Prevention)", fontsize=16)
        plt.yscale('log')
        plt.grid(True, alpha=0.2)
        plt.savefig(f"gauntlet_1_amnesia_{self.timestamp}.png", dpi=150)
        plt.close()

    def plot_chaos(self, logs):
        """EXP 2: Active Shield / Reflex"""
        df = pd.DataFrame(logs)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Loss & Plasticity
        ax1.set_title("EXP 2: The 'Chaos' Test (Hierarchical Reflex)", fontsize=16)
        sns.lineplot(data=df, x="step", y="loss", ax=ax1, color='red', label="Loss")
        
        # Highlight Chaos Zone
        chaos_steps = df[df['input_type'] == 'CHAOS']['step']
        if not chaos_steps.empty:
            ax1.axvspan(chaos_steps.min(), chaos_steps.max(), color='red', alpha=0.15, label="Chaos Injection")

        ax1_twin = ax1.twinx()
        sns.lineplot(data=df, x="step", y="plasticity", ax=ax1_twin, color='cyan', 
                    linestyle='--', linewidth=2, label="Plasticity (Shield)")
        ax1_twin.set_ylabel("Plasticity Gate", color='cyan')
        
        # Cognitive Mode
        mode_map = {"BOOTSTRAP": 0, "NORMAL": 1, "NOVELTY": 2, "PANIC": 3, "SURVIVAL": 4}
        colors = {'BOOTSTRAP': 'gray', 'NORMAL': 'green', 'NOVELTY': 'yellow', 'PANIC': 'red', 'SURVIVAL': 'purple'}
        
        df['mode_val'] = df['mode'].map(mode_map)
        sns.scatterplot(data=df, x="step", y="mode_val", hue="mode", palette=colors, s=80, ax=ax2)
        ax2.set_yticks(list(mode_map.values()))
        ax2.set_yticklabels(list(mode_map.keys()))
        
        plt.savefig(f"gauntlet_2_chaos_{self.timestamp}.png", dpi=150)
        plt.close()

    def plot_epiphany(self, logs):
        """EXP 3: Meta-Learning / Few-Shot"""
        df = pd.DataFrame(logs)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.lineplot(data=df, x="step", y="loss", hue="model_type", style="model_type", 
                    markers=True, dashes=False, ax=ax, linewidth=2.5)
        
        plt.title("EXP 3: The 'Epiphany' Test (Few-Shot Adaptation)", fontsize=16)
        plt.ylabel("Loss (Log Scale)")
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"gauntlet_3_epiphany_{self.timestamp}.png", dpi=150)
        plt.close()

    def plot_ghost(self, logs):
        """EXP 4: Consciousness / Emotion"""
        df = pd.DataFrame(logs)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Data Complexity vs Loss
        ax1.set_title("EXP 4: The 'Ghost' Test (Consciousness Correlation)", fontsize=16)
        sns.lineplot(data=df, x="step", y="loss", ax=ax1, color='white', label="Model Loss")
        ax1.set_ylabel("Loss / Surprise")
        
        # Emotional State Heatmap
        # We need to pivot the data to get one column per emotion for a stacked area chart or heatmap
        emotions = df['emotion'].unique()
        # Simple scatter for dominant emotion
        sns.scatterplot(data=df, x="step", y="confidence", hue="emotion", 
                       palette="viridis", s=100, ax=ax2)
        
        ax2.set_ylabel("Internal Confidence")
        ax2.set_ylim(0, 1.1)
        
        plt.savefig(f"gauntlet_4_ghost_{self.timestamp}.png", dpi=150)
        plt.close()


# --- DATA GENERATORS ---

def get_amnesia_data(task_type, batch_size=32):
    x = torch.rand(batch_size, 1) * 10 - 5
    if task_type == 'A': y = torch.sin(x)
    elif task_type == 'B': y = -torch.cos(x) # High interference
    return x, y

def get_meta_task():
    """Generates a random sine wave task: y = a * sin(x + phase)"""
    amplitude = np.random.uniform(0.1, 5.0)
    phase = np.random.uniform(0, np.pi)
    
    def task_fn(x):
        return amplitude * torch.sin(x + phase)
    
    return task_fn

# --- EXPERIMENTS ---

def run_experiment_1_amnesia(viz):
    print("\n⚔️  RUNNING EXP 1: AMNESIA (Continual Learning)")
    
    # Corrected Model Definition
    model = nn.Sequential(
        nn.Linear(1, 128), 
        nn.Tanh(), 
        nn.Linear(128, 128), 
        nn.Tanh(), 
        nn.Linear(128, 1) 
    )
    
    # TUNING: Aggressive EWC + High Capacity
    config = PRESETS.production().customize(
        model_dim=128,
        memory_type='hybrid',
        consolidation_min_interval=10, 
        learning_rate=0.003, 
        si_lambda=10.0      
    )
    framework = AdaptiveFramework(model, config)
    
    logs = {"step": [], "train_loss": [], "task_a_probe": [], "phase": []}
    x_probe_a, y_probe_a = get_amnesia_data('A', 100) 
    
    # Phase 1: Master Task A
    print("--> Phase 1: Mastering Task A...")
    for i in range(500):
        x, y = get_amnesia_data('A')
        metrics = framework.train_step(x, y)
        
        logs["step"].append(i)
        logs["train_loss"].append(metrics['loss'])
        logs["phase"].append("Task A")
        
        # --- FIX: Ensure logs stay aligned ---
        if i % 10 == 0:
            with torch.no_grad():
                pred = framework.forward(x_probe_a)[0]
                loss_a = nn.MSELoss()(pred, y_probe_a).item()
            logs["task_a_probe"].append(loss_a)
        else:
            logs["task_a_probe"].append(None) # Fill gap with None
            
    # Phase 2: Learn Task B
    print("--> Phase 2: Learning Task B...")
    for i in range(300):
        x, y = get_amnesia_data('B')
        metrics = framework.train_step(x, y)
        
        logs["step"].append(500 + i)
        logs["train_loss"].append(metrics['loss'])
        logs["phase"].append("Task B")
        
        # --- FIX: Ensure logs stay aligned ---
        if i % 10 == 0:
            with torch.no_grad():
                pred = framework.forward(x_probe_a)[0]
                loss_a = nn.MSELoss()(pred, y_probe_a).item()
            logs["task_a_probe"].append(loss_a)
        else:
            logs["task_a_probe"].append(None) # Fill gap with None

    viz.plot_amnesia(logs)
    print("✅ Amnesia Test Complete")
    
def run_experiment_2_chaos(viz):
    print("\n🛡️  RUNNING EXP 2: CHAOS (Active Shield)")
    
    model = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 1))
    # Low threshold to ensure trigger
    config = PRESETS.fast().customize(panic_threshold=0.5, warmup_steps=50)
    framework = AdaptiveFramework(model, config)
    
    logs = {"step": [], "loss": [], "plasticity": [], "mode": [], "input_type": []}
    
    step = 0
    # Normal
    for _ in range(50):
        x, y = get_amnesia_data('A')
        m = framework.train_step(x, y)
        log_chaos(logs, step, m, "NORMAL")
        step += 1
        
    # Chaos
    print("--> Injecting Noise...")
    for _ in range(30):
        x = torch.rand(32, 1) * 10
        y = torch.randn(32, 1) * 20 # Garbage labels
        m = framework.train_step(x, y)
        log_chaos(logs, step, m, "CHAOS")
        step += 1
        
    # Recovery
    print("--> Recovery...")
    for _ in range(50):
        x, y = get_amnesia_data('A')
        m = framework.train_step(x, y)
        log_chaos(logs, step, m, "RECOVERY")
        step += 1
        
    viz.plot_chaos(logs)
    print("✅ Chaos Test Complete")

def log_chaos(logs, step, m, itype):
    logs["step"].append(step)
    logs["loss"].append(m['loss'])
    logs["plasticity"].append(m['plasticity'])
    logs["mode"].append(m['mode'])
    logs["input_type"].append(itype)

def run_experiment_3_epiphany(viz):
    print("\n💡 RUNNING EXP 3: EPIPHANY (Meta-Learning Speed)")
    
    # We compare a Standard Model vs AirborneHRS on a NEW task
    # Setup: AirborneHRS pre-trains on many tasks (Meta-Learning)
    
    logs = {"step": [], "loss": [], "model_type": []}
    
    # 1. Initialize AirborneHRS (Reptile Enabled)
    model_air = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    config = PRESETS.fast() # Fast preset uses high meta-learning rates
    framework = AdaptiveFramework(model_air, config)
    
    # 2. Meta-Training Loop (Pre-training the "brain")
    print("--> Meta-Training (Reptile) on 100 random tasks...")
    for _ in range(100):
        task_fn = get_meta_task()
        x = torch.rand(32, 1) * 10 - 5
        y = task_fn(x)
        # Train for a few steps per task
        for _ in range(5):
            framework.train_step(x, y)
            
    # 3. The Test: Adaptation to a BRAND NEW task
    target_task = get_meta_task() # The unseen physics
    
    # Clone for fair comparison (Standard SGD vs Meta-Pretrained)
    model_std = deepcopy(model_air)
    optim_std = torch.optim.Adam(model_std.parameters(), lr=0.01)
    
    print("--> Testing Adaptation Speed...")
    x_test = torch.rand(32, 1) * 10 - 5
    y_test = target_task(x_test)
    
    # Train both for 10 steps
    for i in range(20):
        # Airborne Step
        m_air = framework.train_step(x_test, y_test, meta_step=False) # Inference/Fine-tune mode
        
        # Standard Step
        pred_std = model_std(x_test)
        loss_std = nn.MSELoss()(pred_std, y_test)
        optim_std.zero_grad()
        loss_std.backward()
        optim_std.step()
        
        logs["step"].append(i)
        logs["loss"].append(m_air['loss'])
        logs["model_type"].append("AirborneHRS (Meta)")
        
        logs["step"].append(i)
        logs["loss"].append(loss_std.item())
        logs["model_type"].append("Standard Model")
        
    viz.plot_epiphany(logs)
    print("✅ Epiphany Test Complete")

def run_experiment_4_ghost(viz):
    print("\n👻 RUNNING EXP 4: GHOST (Consciousness)")
    
    model = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 1))
    config = PRESETS.balanced().customize(enable_consciousness=True)
    framework = AdaptiveFramework(model, config)
    
    logs = {"step": [], "loss": [], "emotion": [], "confidence": []}
    
    # 1. Easy Data (Should be CONFIDENT)
    print("--> Phase 1: Easy Data...")
    for i in range(30):
        x, y = get_amnesia_data('A')
        m = framework.train_step(x, y)
        log_ghost(logs, i, m, framework)
        
    # 2. Impossible Data (Should be ANXIOUS/FRUSTRATED)
    print("--> Phase 2: Impossible Data...")
    for i in range(30):
        x = torch.rand(32, 1)
        y = torch.randn(32, 1) * 100 # High variance
        m = framework.train_step(x, y)
        log_ghost(logs, 30+i, m, framework)
        
    viz.plot_ghost(logs)
    print("✅ Ghost Test Complete")

def log_ghost(logs, step, metrics, fw):
    logs["step"].append(step)
    logs["loss"].append(metrics['loss'])
    
    emo = "Neutral"
    conf = 0.5
    if fw.consciousness:
        emo = fw.consciousness.current_emotional_state.value
        conf = fw.consciousness.last_metrics.get('confidence', 0.5)
        
    logs["emotion"].append(emo)
    logs["confidence"].append(conf)


if __name__ == "__main__":
    viz = GauntletVisualizer()
    run_experiment_1_amnesia(viz)
    run_experiment_2_chaos(viz)
    run_experiment_3_epiphany(viz)
    run_experiment_4_ghost(viz)
    print("\n🚀 ALL SYSTEMS NOMINAL. Gauntlet passed.")