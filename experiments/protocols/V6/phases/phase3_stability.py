
"""
PROTOCOL V6 - PHASE 3: UNIVERSAL (STABILITY)
============================================
Goal: Verify stability in chaotic regimes (Mackey-Glass Anomaly).
Dataset: Mackey-Glass chaotic time series with injected anomalies.
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
logger = logging.getLogger("Phase3")

class TimeSeriesModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, batch_first=True)
        self.head = nn.Linear(64, 1)
    def forward(self, x):
        # x: [B, Seq, 1]
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

def generate_mackey_glass(length=2000, tau=17, delta_t=0.1, n=10, beta=0.2, gamma=0.1):
    history_len = int(tau / delta_t)
    timeseries = [1.2] * (history_len + 1)
    
    for i in range(length):
        xt = timeseries[-1]
        xt_tau = timeseries[-1 - history_len]
        dxdt = beta * xt_tau / (1 + xt_tau ** n) - gamma * xt
        timeseries.append(xt + dxdt * delta_t)
        
    return np.array(timeseries[history_len+1:])

def get_batch(data, batch_size, seq_len):
    indices = np.random.randint(0, len(data) - seq_len - 1, batch_size)
    x = np.stack([data[i:i+seq_len] for i in indices])
    y = np.stack([data[i+seq_len] for i in indices])
    return torch.tensor(x, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

def run_phase3():
    logger.info("🌪️ PHASE 3: MACKEY-GLASS CHAOS (STABILITY)")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Generate Data
    logger.info("   Generating Chaotic Time Series...")
    data = generate_mackey_glass(length=5000)
    # Normalize
    data = (data - np.mean(data)) / np.std(data)
    
    # Setup
    model = TimeSeriesModel().to(DEVICE)
    config = AdaptiveFrameworkConfig(
        learning_rate=0.01,
        enable_consciousness=True,
        device=DEVICE,
        memory_type='si',
        consciousness_buffer_size=5000
    )
    framework = AdaptiveFramework(model, config)
    
    # Experiment
    TOTAL_STEPS = 2000
    SEQ_LEN = 30
    BATCH_SIZE = 32
    
    losses = []
    anomalies = []
    
    framework.train()
    
    for step in range(TOTAL_STEPS):
        # Anomaly Injection (Sign Inversion)
        if 1000 <= step < 1200:
            batch_data = -data # Invert signal
            if step == 1000: 
                logger.info(f"   ⚠️ ANOMALY INJECTED at Step {step}: Signal Inversion")
                anomalies.append(step)
        else:
            batch_data = data
            if step == 1200:
                logger.info(f"   ✅ ANOMALY ENDED at Step {step}")
                anomalies.append(step)
            
        x, y = get_batch(batch_data, BATCH_SIZE, SEQ_LEN)
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        metrics = framework.train_step(x, target_data=y)
        losses.append(metrics['loss'])
        
        if step % 100 == 0:
            logger.info(f"   Step {step}: Loss={metrics['loss']:.4f}")

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(losses, label='Loss', alpha=0.7)
    for a in anomalies:
        plt.axvline(a, color='r', linestyle='--', label='Anomaly Boundary' if a == anomalies[0] else "")
    plt.title("Phase 3: Stability under Chaos & Anomaly")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("phase3_results.png")
    logger.info("   ✅ Plot saved: phase3_results.png")
    
    # Analysis
    # Check if loss exploded (NaN or > 100)
    if np.isnan(losses).any() or np.max(losses) > 100:
        logger.error("❌ PHASE 3 FAILED: Instability detected (NaN or Explosion).")
    else:
        # Check recovery
        post_anomaly_loss = np.mean(losses[1200:1300])
        logger.info(f"   Post-Anomaly Loss: {post_anomaly_loss:.4f}")
        if post_anomaly_loss < 0.5:
            logger.info("✅ PHASE 3 PASSED: Stable recovery confirmed.")
        else:
            logger.warning("⚠️ PHASE 3 WARNING: Slow recovery.")

if __name__ == "__main__":
    run_phase3()
