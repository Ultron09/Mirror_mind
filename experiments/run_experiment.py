import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig
from airbornehrs.meta_controller import MetaController, MetaControllerConfig
import random

# Enable safety checks
torch.autograd.set_detect_anomaly(True)

def generate_task_data(phase, batch_size=32, seq_len=10, dim=32):
    """
    Generates data for different 'concepts' to simulate drift.
    """
    x = torch.randn(batch_size, seq_len, dim)
    
    if phase == "IDENTITY":
        # Task 1: Output = Input (Easy)
        y = x.clone()
        noise = 0.0
        
    elif phase == "NEGATION":
        # Task 2: Output = -Input (Medium - Pattern Reversal)
        y = -x.clone()
        noise = 0.0
        
    elif phase == "SCRAMBLE":
        # Task 3: Output = Scaled + Shifted (Hard - Distribution Shift)
        # y = 2x + 1
        y = (x.clone() * 2.0) + 1.0
        noise = 0.0
        
    elif phase == "CHAOS":
        # Task 4: High Noise Environment (Stress Test)
        y = x.clone()
        noise = 0.5 # Massive noise injection
    
    # Add observation noise
    if noise > 0:
        x = x + torch.randn_like(x) * noise
        
    return x, y

def run_extreme_stress_test():
    # ---------------------------------------------------------
    # 1. HARDCORE CONFIGURATION
    # ---------------------------------------------------------
    print("🔥 INITIALIZING EXTREME STRESS TEST PROTOCOL...")
    
    # Framework: Capable but constrained to force adaptation
    framework_config = AdaptiveFrameworkConfig(
        model_dim=32, 
        num_layers=4,          # Deeper network
        learning_rate=0.0001,  # Start VERY slow to prove dynamic LR works
        adaptation_threshold=0.02,
        evaluation_frequency=2
    )
    
    # Controller: Aggressive limits
    meta_config = MetaControllerConfig(
        base_lr=0.0001,
        max_lr=0.05,       # Allow massive spikes (500x base)
        min_lr=1e-6,
        curriculum_start_difficulty=0.1
    )
    
    framework = AdaptiveFramework(framework_config, device='cpu')
    controller = MetaController(framework, config=meta_config)
    
    # ---------------------------------------------------------
    # 2. THE GAUNTLET (400 Steps per Phase)
    # ---------------------------------------------------------
    history = {
        'loss': [],
        'lr': [],
        'uncertainty': [],
        'grad_norm': []
    }
    
    phases = [
        ("IDENTITY", 200),
        ("NEGATION", 200),
        ("SCRAMBLE", 200),
        ("CHAOS",    200)
    ]
    
    total_steps = sum(p[1] for p in phases)
    current_step = 0
    
    print(f"📉 Baseline LR: {meta_config.base_lr}")
    print(f"📈 Max Allowed LR: {meta_config.max_lr}")
    print("-" * 60)

    for phase_name, steps in phases:
        print(f"⚡ PHASE START: {phase_name} (Steps {current_step}-{current_step+steps})")
        
        for i in range(steps):
            # 1. Get Data
            x, y = generate_task_data(phase_name)
            
            # 2. Train Step
            metrics = framework.train_step(x, y)
            
            # 3. Adapt (The Magic)
            adaptation = controller.adapt(
                loss=metrics['loss'],
                performance_metrics={'loss_improvement': 0.0} # Let analyzer figure it out
            )
            
            # 4. Log Telemetry
            history['loss'].append(metrics['loss'])
            history['uncertainty'].append(metrics.get('uncertainty_mean', 0))
            history['lr'].append(adaptation['learning_rate'])
            
            # Extract gradient norm from analyzer stats if available
            grad_stats = adaptation.get('gradient_stats', {})
            history['grad_norm'].append(grad_stats.get('mean_norm', 0))
            
            current_step += 1
            
            if i == 10: print(f"   ...First 10 steps survived.")

    # ---------------------------------------------------------
    # 3. ADVANCED ANALYSIS & VISUALIZATION
    # ---------------------------------------------------------
    print("\n📊 GENERATING RESEARCH DASHBOARD...")
    
    # Smooth curves for pretty plotting
    def smooth(data, window=5):
        return np.convolve(data, np.ones(window)/window, mode='valid')

    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
    
    # PANEL 1: Loss & Phases
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(history['loss'], color='blue', alpha=0.3, label='Raw Loss')
    ax1.plot(smooth(history['loss']), color='darkblue', linewidth=2, label='Smoothed Loss')
    ax1.set_yscale('log')
    ax1.set_ylabel('Loss (Log Scale)', fontsize=12, fontweight='bold')
    ax1.set_title('MirrorMind Response to Multi-Stage Concept Drift', fontsize=14, fontweight='bold')
    ax1.grid(True, which='both', alpha=0.2)
    ax1.legend(loc='upper right')
    
    # Draw Phase Lines
    boundary = 0
    colors = ['green', 'red', 'orange', 'purple']
    for (name, steps), color in zip(phases, colors):
        ax1.axvline(x=boundary, color=color, linestyle='--', alpha=0.5)
        ax1.text(boundary + 10, ax1.get_ylim()[1]*0.5, name, color=color, fontweight='bold', rotation=90)
        boundary += steps

    # PANEL 2: The Stabilizer (Learning Rate)
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(history['lr'], color='red', linewidth=2, label='Adaptive LR')
    ax2.set_ylabel('Learning Rate', color='red', fontsize=12, fontweight='bold')
    ax2.fill_between(range(len(history['lr'])), 0, history['lr'], color='red', alpha=0.1)
    ax2.set_ylim(0, meta_config.max_lr * 1.1)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # PANEL 3: Internal State (Uncertainty & Gradients)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    
    # Gradient Norm (Left Axis)
    l1 = ax3.plot(smooth(history['grad_norm']), color='purple', label='Gradient Norm')
    ax3.set_ylabel('Grad Norm', color='purple', fontsize=12, fontweight='bold')
    
    # Uncertainty (Right Axis)
    ax3b = ax3.twinx()
    l2 = ax3b.plot(smooth(history['uncertainty']), color='orange', linestyle='--', label='Uncertainty (LogVar)')
    ax3b.set_ylabel('Uncertainty', color='orange', fontsize=12, fontweight='bold')
    
    # Combined Legend
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax3.legend(lns, labs, loc='upper right')
    
    ax3.set_xlabel('Global Training Steps', fontsize=12)

    # Save
    plt.savefig("proof_extreme.png", dpi=150, bbox_inches='tight')
    print("✅ Dashboard saved to: proof_extreme.png")
    
    # ---------------------------------------------------------
    # 4. FINAL REPORT
    # ---------------------------------------------------------
    print("\n🏆 RESULTS SUMMARY:")
    boundary = 0
    for i, (name, steps) in enumerate(phases):
        start = boundary
        end = boundary + steps
        
        # Calculate max reflex spike in this phase
        phase_lrs = history['lr'][start:end]
        max_spike = max(phase_lrs)
        spike_ratio = max_spike / meta_config.base_lr
        
        print(f"   [{name}] Max LR Spike: {max_spike:.5f} ({spike_ratio:.1f}x Base)")
        boundary += steps

if __name__ == "__main__":
    run_extreme_stress_test()