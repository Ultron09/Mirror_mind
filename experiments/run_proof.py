import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig
from airbornehrs.meta_controller import MetaController, MetaControllerConfig

# SAFETY: Detect any NaN gradients immediately
torch.autograd.set_detect_anomaly(True)

def run_proof_of_concept():
    # 1. Framework Config
    framework_config = AdaptiveFrameworkConfig(
        model_dim=32, 
        num_layers=2, 
        learning_rate=0.0005,  # Start SLOW to allow room for spike
        adaptation_threshold=0.05,
        evaluation_frequency=5
    )
    
    # 2. MetaController Config 
    meta_config = MetaControllerConfig(
        base_lr=0.0005,
        max_lr=0.02,     # Higher ceiling so we can see the spike
        min_lr=0.0001
    )
    
    # 3. Init
    framework = AdaptiveFramework(framework_config, device='cpu')
    controller = MetaController(framework, config=meta_config)
    
    losses = []
    lrs = []
    
    print("🚀 Starting Final Verified Experiment...")
    
    # ---------------------------------------------------------
    # PHASE 1: Identity Task
    # ---------------------------------------------------------
    print("Phase 1: Learning Identity (Copy)...")
    for step in range(250): # Longer phase 1 to ensure low LR convergence
        x = torch.randn(32, 10, 32) 
        y = x.clone()
        
        metrics = framework.train_step(x, y)
        adaptation = controller.adapt(metrics['loss'])
        
        losses.append(metrics['loss'])
        lrs.append(adaptation['learning_rate'])

    # ---------------------------------------------------------
    # PHASE 2: Negation Task (The Shift)
    # ---------------------------------------------------------
    print("Phase 2: SUDDEN TASK SHIFT (Target -> Invert)...")
    for step in range(250, 500):
        x = torch.randn(32, 10, 32)
        y = -x.clone()
        
        metrics = framework.train_step(x, y)
        adaptation = controller.adapt(metrics['loss'])
        
        losses.append(metrics['loss'])
        lrs.append(adaptation['learning_rate'])

    # ---------------------------------------------------------
    # ANALYSIS
    # ---------------------------------------------------------
    base_lr = np.mean(lrs[200:250]) # Avg LR before shift
    peak_lr = max(lrs[250:300])     # Max LR after shift
    
    print(f"\nFinal Loss: {losses[-1]:.6f}")
    print(f"LR converged at: {base_lr:.5f}")
    print(f"LR spike max:    {peak_lr:.5f}")
    
    if peak_lr > base_lr * 1.5: # Expect significant spike
        print("✅ SUCCESS: Reflex response confirmed!")
    else:
        print("⚠️ NOTE: Reflex response weak.")

    # Plotting
    try:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(losses, 'b-', alpha=0.6, label='Loss')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss', color='b')
        ax1.set_yscale('log')
        
        ax2 = ax1.twinx()
        ax2.plot(lrs, 'r--', linewidth=2, label='Adaptive LR')
        ax2.set_ylabel('Learning Rate', color='r')
        
        plt.title("MirrorMind: Adaptive Response to Concept Drift")
        plt.axvline(x=250, color='k', linestyle=':', label='Concept Drift')
        plt.savefig("proof_of_adaptation_v3.png")
        print("Saved plot to proof_of_adaptation_v3.png")
    except Exception as e:
        print(f"Plotting error: {e}")

if __name__ == "__main__":
    run_proof_of_concept()