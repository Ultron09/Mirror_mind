
"""
PROTOCOL V6 - PHASE 1: INTEGRITY (PLASTICITY)
=============================================
Goal: Verify rapid adaptation to drifting distributions (Drifting Sinusoid).
Dataset: y = A * sin(x + phi), where phi shifts every 1000 steps.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import logging

# Path Setup
# Path Setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger("Phase1")

class SinusoidModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

def generate_batch(batch_size, phase_shift):
    x = torch.rand(batch_size, 1) * 10 - 5 # Range [-5, 5]
    y = torch.sin(x + phase_shift)
    return x, y

def run_phase1():
    logger.info("🌊 PHASE 1: DRIFTING SINUSOID (PLASTICITY)")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Setup
    model = SinusoidModel().to(DEVICE)
    config = AdaptiveFrameworkConfig(
        learning_rate=0.01,
        enable_consciousness=True,
        device=DEVICE,
        memory_type='si',
        dream_interval=50,
        consciousness_buffer_size=5000
    )
    framework = AdaptiveFramework(model, config)
    
    # Experiment Params
    TOTAL_STEPS = 5000
    SHIFT_INTERVAL = 1000
    BATCH_SIZE = 32
    
    current_phase = 0.0
    losses = []
    shifts = []
    
    framework.train()
    
    for step in range(TOTAL_STEPS):
        # Drift Logic
        if step > 0 and step % SHIFT_INTERVAL == 0:
            current_phase += np.pi / 2 # 90 degree shift
            shifts.append(step)
            logger.info(f"   ⚠️ DRIFT EVENT at Step {step}: Phase += 90 deg")
            
            # Signal Consciousness (Optional, simulates surprise)
            # In a real scenario, high loss should trigger this automatically.
            
        # Data
        x, y = generate_batch(BATCH_SIZE, current_phase)
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        # Train
        metrics = framework.train_step(x, target_data=y)
        losses.append(metrics['loss'])
        
        if step % 100 == 0:
            logger.info(f"   Step {step}: Loss={metrics['loss']:.4f} | Phase={current_phase:.2f}")

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(losses, label='Loss', alpha=0.6)
    for s in shifts:
        plt.axvline(s, color='r', linestyle='--', label='Drift Event' if s == shifts[0] else "")
    plt.yscale('log')
    plt.title("Phase 1: Adaptation to Drifting Sinusoid")
    plt.xlabel("Step")
    plt.ylabel("Loss (Log Scale)")
    plt.legend()
    plt.savefig("phase1_results.png")
    logger.info("   ✅ Plot saved: phase1_results.png")
    
    # Analysis
    # Check recovery speed after drift
    recovery_window = 100
    success = True
    for s in shifts:
        post_drift_losses = losses[s:s+recovery_window]
        if not post_drift_losses: continue
        avg_loss = np.mean(post_drift_losses)
        logger.info(f"   Drift at {s}: Avg Loss (next {recovery_window} steps) = {avg_loss:.4f}")
        if avg_loss > 0.1: # Threshold depends on difficulty
            success = False
            logger.warning("   ⚠️ Slow adaptation detected.")
            
    if success:
        logger.info("✅ PHASE 1 PASSED: Rapid adaptation confirmed.")
    else:
        logger.warning("⚠️ PHASE 1 WARNING: Adaptation might be too slow.")

if __name__ == "__main__":
    run_phase1()
