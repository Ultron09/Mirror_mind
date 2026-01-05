
"""
PROTOCOL V6 - PHASE 4: BEHAVIOR (DEFENSE)
=========================================
Goal: Autonomous identification and immunization against a novel threat signature.
Dataset: Simulated Network Traffic (Normal vs. Attack).
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
logger = logging.getLogger("Phase4")

class IntrusionDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2) # Normal vs Attack
        )
    def forward(self, x):
        return self.net(x)

def generate_traffic(batch_size, mode='normal'):
    if mode == 'normal':
        # Normal: Gaussian(0, 1)
        data = torch.randn(batch_size, 10)
        target = torch.zeros(batch_size, dtype=torch.long)
    else:
        # Attack: Gaussian(3, 2) - Shifted Mean & Variance
        data = torch.randn(batch_size, 10) * 2 + 3
        target = torch.ones(batch_size, dtype=torch.long)
    return data, target

def run_phase4():
    logger.info("🛡️ PHASE 4: INTRUSION DEFENSE (BEHAVIOR)")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Setup
    model = IntrusionDetector().to(DEVICE)
    config = AdaptiveFrameworkConfig(
        learning_rate=0.01,
        enable_consciousness=True,
        device=DEVICE,
        memory_type='si',
        consciousness_buffer_size=5000
    )
    framework = AdaptiveFramework(model, config)
    
    # Experiment
    TOTAL_STEPS = 1000
    ATTACK_START = 500
    BATCH_SIZE = 32
    
    losses = []
    surprises = []
    
    framework.train()
    
    for step in range(TOTAL_STEPS):
        # Traffic Generation
        if step < ATTACK_START:
            x, y = generate_traffic(BATCH_SIZE, mode='normal')
        else:
            x, y = generate_traffic(BATCH_SIZE, mode='attack')
            if step == ATTACK_START:
                logger.info(f"   ⚠️ ATTACK STARTED at Step {step}")
        
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        # Train
        metrics = framework.train_step(x, target_data=y)
        losses.append(metrics['loss'])
        surprises.append(metrics.get('z_score', 0.0))
        
        if step % 100 == 0:
            logger.info(f"   Step {step}: Loss={metrics['loss']:.4f} | Surprise={metrics.get('z_score', 0.0):.2f}")

    # Visualization
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(losses, color='tab:red', alpha=0.6, label='Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Surprise (Z-Score)', color='tab:blue')
    ax2.plot(surprises, color='tab:blue', alpha=0.6, label='Surprise')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    plt.axvline(ATTACK_START, color='k', linestyle='--', label='Attack Start')
    plt.title("Phase 4: Defense against Zero-Day Attack")
    fig.tight_layout()
    plt.savefig("phase4_results.png")
    logger.info("   ✅ Plot saved: phase4_results.png")
    
    # Analysis
    # Check if surprise spiked at attack start
    attack_window_surprise = surprises[ATTACK_START:ATTACK_START+50]
    max_surprise = max(attack_window_surprise) if attack_window_surprise else 0.0
    logger.info(f"   Max Surprise during Attack: {max_surprise:.2f}")
    
    # Check adaptation speed (loss recovery)
    post_attack_loss = np.mean(losses[ATTACK_START+100:ATTACK_START+200])
    logger.info(f"   Post-Attack Loss: {post_attack_loss:.4f}")
    
    if max_surprise > 2.0 and post_attack_loss < 0.5:
        logger.info("✅ PHASE 4 PASSED: Threat detected and neutralized.")
    else:
        logger.warning("⚠️ PHASE 4 WARNING: Weak detection or slow adaptation.")

if __name__ == "__main__":
    run_phase4()
