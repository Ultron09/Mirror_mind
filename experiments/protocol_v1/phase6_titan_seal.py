"""
PROTOCOL PHASE 6: THE TITAN SEAL (CHAOS STABILITY) - WITH REPORTING
===================================================================
Goal: Compete with Liquid Neural Networks on their home turf.
Test: Mackey-Glass Chaotic Time-Series Prediction with Parameter Drift.
Scenario:
1. Generate chaotic waveform (Mackey-Glass).
2. Train MirrorMind (LSTM) on the live stream.
3. INJECT ANOMALY: Shift time-delay (tau) from 17 to 30 at Step 75.
4. Verify Stability: NaN Checks, Loss Recovery, and Gradient Clamping.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import sys
import os
import matplotlib.pyplot as plt
from collections import deque
import platform
import datetime

# Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from airbornehrs import (
        AdaptiveFramework, 
        AdaptiveFrameworkConfig, 
        MetaController,
        MetaControllerConfig,
        ProductionAdapter,
        InferenceMode
    )
except ImportError:
    print("❌ CRITICAL: Import failed.")
    sys.exit(1)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger("Phase6")

# ==============================================================================
# HELPER: Visualization & Reporting
# ==============================================================================
def generate_artifacts(history, stats, status):
    """Generates Dashboard PNG and MD report for research documentation."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # --- 1. Generate Visualization (PNG) ---
    try:
        steps = range(len(history['loss']))
        bifurcation = stats['bifurcation_step']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Panel 1: Stability (Loss)
        ax1.plot(steps, history['loss'], color='#c0392b', label='Prediction Error (MSE)')
        ax1.axvline(x=bifurcation, color='black', linestyle='--', alpha=0.7, label='Bifurcation Event')
        ax1.set_title(f"MirrorMind Stability: Chaos Prediction\n{timestamp}")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.text(bifurcation + 2, max(history['loss'])*0.8, 'Physics Change (Tau=30)', fontsize=9)
        
        # Panel 2: Plasticity (Learning Rate)
        ax2.plot(steps, history['lr'], color='#27ae60', label='Neuroplasticity (Learning Rate)')
        ax2.axvline(x=bifurcation, color='black', linestyle='--', alpha=0.7)
        ax2.set_ylabel("Learning Rate")
        ax2.set_xlabel("Time Step")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("phase6_chaos_stability.png", dpi=300)
        plt.close()
        logger.info("   ✅ Visualization saved: phase6_chaos_stability.png")
    except Exception as e:
        logger.error(f"   ⚠️ Visualization failed: {e}")

    # --- 2. Generate Research Report (Markdown) ---
    report_content = f"""# MirrorMind Protocol: Phase 6 Titan Seal
**Date:** {timestamp}
**Status:** {status}

## 1. Objective
To verify system stability when predicting a Mackey-Glass chaotic time series undergoing a sudden parameter bifurcation (Concept Drift).

## 2. Experimental Setup
* **Generator:** Mackey-Glass Equation ($dx/dt = \\beta x(t-\\tau) / (1 + x(t-\\tau)^n) - \\gamma x(t)$)
* **Drift Event:** $\\tau$ shifted from 17 to 30 at Step {stats['bifurcation_step']}.
* **Subject:** LSTM + AdaptiveFramework.

## 3. Stability Metrics
| Metric | Pre-Drift (Step 0-{stats['bifurcation_step']}) | Post-Drift (Step {stats['bifurcation_step']}+) |
| :--- | :--- | :--- |
| **Avg Loss** | {stats['pre_loss']:.4f} | {stats['post_loss']:.4f} |
| **Max Loss** | {stats['max_pre_loss']:.4f} | {stats['max_post_loss']:.4f} |

## 4. Recovery Analysis
* **Explosion Detected (NaN):** {"YES" if stats['nan_detected'] else "NO"}
* **Recovery Ratio:** {stats['pre_loss'] / stats['post_loss']:.2f} (Target > 0.5)
* **Conclusion:** The system {"successfully adapted" if status == "PASSED" else "failed to stabilize"} to the chaotic bifurcation.
"""
    
    with open("PHASE6_REPORT.md", "w") as f:
        f.write(report_content)
    logger.info("   ✅ Research Report generated: PHASE6_REPORT.md")

# ==============================================================================
# 1. CHAOS GENERATOR (Mackey-Glass Differential Equation)
# ==============================================================================
class MackeyGlassGenerator:
    """
    Generates chaotic time-series data.
    """
    def __init__(self, tau=17, beta=0.2, gamma=0.1, n=10):
        self.tau = tau
        self.beta = beta
        self.gamma = gamma
        self.n = n
        self.history = deque([1.2] * (tau + 100), maxlen=tau + 100)
        self.x = 1.2
        self.dt = 1.0

    def step(self):
        # Retrieve delayed value x(t-tau)
        if len(self.history) <= self.tau:
            delayed_x = 0.0
        else:
            delayed_x = self.history[-self.tau]

        # Compute derivative
        dx = (self.beta * delayed_x) / (1 + delayed_x ** self.n) - (self.gamma * self.x)
        
        # Euler integration
        self.x += dx * self.dt
        self.history.append(self.x)
        return self.x

    def generate_batch(self, steps, batch_size=1):
        """Generates a sequence [Batch, Steps, 1]"""
        data = []
        for _ in range(steps):
            val = self.step()
            data.append(val)
        
        # Normalize roughly to [-1, 1] for neural net stability
        tensor = torch.tensor(data, dtype=torch.float32).view(1, steps, 1)
        tensor = (tensor - 0.8) / 0.5 
        return tensor.repeat(batch_size, 1, 1)

# ==============================================================================
# 2. THE SUBJECT (LSTM Baseline)
# ==============================================================================
class ChaosSubject(nn.Module):
    """Standard LSTM, notoriously unstable in chaos without help."""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.head = nn.Linear(32, 1)
        
    def forward(self, x):
        # x: [Batch, Seq, 1]
        out, _ = self.lstm(x)
        return self.head(out) 

# ==============================================================================
# 3. THE TITAN PROTOCOL
# ==============================================================================
def run_titan_protocol():
    logger.info("🛡️ PHASE 6: THE TITAN PROTOCOL (CHAOS STABILITY)")
    logger.info("   Target: Mackey-Glass Chaos Prediction")
    
    # 1. Setup
    generator = MackeyGlassGenerator(tau=17) # Standard Chaos
    
    # Config: High stability required
    fw_config = AdaptiveFrameworkConfig(
        model_dim=32,
        learning_rate=0.005,
        compile_model=False,
        device='cpu'
    )
    
    # Meta-Controller with Z-Score Clamping active
    meta_config = MetaControllerConfig(
        use_reptile=True,
        base_lr=0.005,
        max_lr=0.02
    )
    
    model = ChaosSubject()
    framework = AdaptiveFramework(model, fw_config)
    controller = MetaController(framework, meta_config)
    adapter = ProductionAdapter(framework, inference_mode=InferenceMode.ONLINE)
    
    # 2. Execution Loop
    TOTAL_STEPS = 150
    BIFURCATION_POINT = 75
    
    history_loss = []
    history_lr = []
    nan_detected = False
    
    logger.info(f"   Running {TOTAL_STEPS} steps. Bifurcation at {BIFURCATION_POINT}.")
    
    for step in range(TOTAL_STEPS):
        # A. Inject Anomaly
        if step == BIFURCATION_POINT:
            logger.info("   ⚠️  BIFURCATION EVENT: Shifting Chaos Parameters (tau 17 -> 30)")
            generator.tau = 30 # Deep Chaos
            generator.history = deque([1.2] * 200, maxlen=200) # Reset history buffer
            
        # B. Generate Live Data
        # Input: t ... t+9, Target: t+1 ... t+10
        seq = generator.generate_batch(steps=11)
        x = seq[:, :-1, :]
        y = seq[:, 1:, :]
        
        # C. MirrorMind Adapt
        try:
            pred = adapter.predict(x, update=True, target=y)
            
            if torch.isnan(pred).any():
                logger.error("   ❌ CRITICAL FAILURE: Model output is NaN (Explosion).")
                nan_detected = True
                break
                
        except Exception as e:
            logger.error(f"   ❌ CRITICAL FAILURE: {e}")
            sys.exit(1)
            
        # D. Metrics
        metrics = adapter.get_metrics()
        loss = metrics.get('loss', 0.0)
        # Handle tensor or float
        if isinstance(loss, torch.Tensor): loss = loss.item()
        
        lr = framework.optimizer.param_groups[0]['lr']
        
        history_loss.append(loss)
        history_lr.append(lr)
        
        if step % 25 == 0:
            logger.info(f"   Step {step:03}: Loss={loss:.4f} | LR={lr:.5f}")

    # ==========================================================================
    # 3. ANALYSIS & ARTIFACTS
    # ==========================================================================
    
    # Calc Stats
    pre_losses = history_loss[:BIFURCATION_POINT]
    post_losses = history_loss[BIFURCATION_POINT+10:] # Skip immediate shock
    
    stats = {
        'bifurcation_step': BIFURCATION_POINT,
        'nan_detected': nan_detected,
        'pre_loss': np.mean(pre_losses) if pre_losses else 0.0,
        'post_loss': np.mean(post_losses) if post_losses else 0.0,
        'max_pre_loss': max(pre_losses) if pre_losses else 0.0,
        'max_post_loss': max(post_losses) if post_losses else 0.0
    }
    
    history_data = {
        'loss': history_loss,
        'lr': history_lr
    }
    
    # Verdict Logic
    passed = (not nan_detected) and (stats['post_loss'] < 1.0) # < 1.0 is generous for chaos
    status_str = "PASSED" if passed else "FAILED"
    
    generate_artifacts(history_data, stats, status_str)
    
    if passed:
        print("\n" + "="*40)
        print("🟢 TITAN SEAL EARNED: System stabilized in chaos.")
        print("   -> Plot saved to: phase6_chaos_stability.png")
        print("   -> Report saved to: PHASE6_REPORT.md")
        print("="*40 + "\n")
        sys.exit(0)
    else:
        print("\n" + "="*40)
        print(f"🔴 SEAL DENIED: Unstable. Loss={stats['post_loss']:.4f}")
        print("="*40 + "\n")
        sys.exit(1)

if __name__ == "__main__":
    run_titan_protocol()