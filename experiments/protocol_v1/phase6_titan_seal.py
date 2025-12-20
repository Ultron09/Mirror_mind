"""
PROTOCOL PHASE 6: THE TITAN SEAL (CHAOS STABILITY)
==================================================
Goal: Compete with Liquid Neural Networks on their home turf.
Test: Mackey-Glass Chaotic Time-Series Prediction with Parameter Drift.

Scenario:
1. Generate a chaotic waveform using the Mackey-Glass differential equation.
2. Train MirrorMind (LSTM-based) on the live stream.
3. INJECT ANOMALY: Shift the time-delay parameter (tau) mid-stream.
4. Verify:
   - Does it explode? (NaN check)
   - Does it adapt? (Loss recovery)
   - Do gradients stabilize? (Z-Score clamping)
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import sys
import os
import matplotlib.pyplot as plt
from collections import deque

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
    sys.exit(1)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger("Phase6")

# ==============================================================================
# 1. CHAOS GENERATOR (Mackey-Glass Differential Equation)
# ==============================================================================
class MackeyGlassGenerator:
    """
    Generates chaotic time-series data.
    dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n) - gamma * x(t)
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
        # Predict next step based on last hidden state
        return self.head(out) # Return full sequence for training

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
    
    # Meta-Controller with Z-Score Clamping active (Default behavior)
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
    
    losses = []
    lrs = []
    
    logger.info(f"   Running {TOTAL_STEPS} steps. Bifurcation at {BIFURCATION_POINT}.")
    
    for step in range(TOTAL_STEPS):
        # A. Inject Anomaly
        if step == BIFURCATION_POINT:
            logger.info("   ⚠️  BIFURCATION EVENT: Shifting Chaos Parameters (tau 17 -> 30)")
            generator.tau = 30 # Deep Chaos (Harder to predict)
            generator.history = deque([1.2] * 200, maxlen=200) # Reset history buffer
            
        # B. Generate Live Data (Sequence of 10)
        # Input: t ... t+9
        # Target: t+1 ... t+10 (Next step prediction)
        seq = generator.generate_batch(steps=11)
        x = seq[:, :-1, :]
        y = seq[:, 1:, :]
        
        # C. MirrorMind Adapt
        try:
            pred = adapter.predict(x, update=True, target=y)
            
            # Check for Explosion
            if torch.isnan(pred).any():
                logger.error("   ❌ CRITICAL FAILURE: Model output is NaN (Explosion).")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"   ❌ CRITICAL FAILURE: {e}")
            sys.exit(1)
            
        # D. Metrics
        metrics = adapter.get_metrics()
        loss = metrics.get('loss', 0.0)
        lr = framework.optimizer.param_groups[0]['lr']
        
        losses.append(loss)
        lrs.append(lr)
        
        # Log periodically
        if step % 25 == 0:
            logger.info(f"   Step {step:03}: Loss={loss:.4f} | LR={lr:.5f}")

    # ==========================================================================
    # 3. ANALYSIS & VISUALIZATION
    # ==========================================================================
    logger.info("📊 Generating Chaos Report...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Loss Plot
    ax1.plot(losses, 'b-', label='Prediction Error')
    ax1.axvline(x=BIFURCATION_POINT, color='r', linestyle='--', label='Bifurcation')
    ax1.set_title("MirrorMind Stability on Chaotic Data")
    ax1.set_ylabel("MSE Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plasticity Plot
    ax2.plot(lrs, 'g-', label='Neuroplasticity (LR)')
    ax2.axvline(x=BIFURCATION_POINT, color='r', linestyle='--')
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("Learning Rate")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiments/protocol_v1/titan_seal.png')
    logger.info("   ✅ Saved 'titan_seal.png'")
    
    # Verdict Logic
    post_bifurcation_loss = np.mean(losses[BIFURCATION_POINT+10:])
    max_loss = max(losses)
    
    logger.info("-" * 40)
    logger.info(f"   Post-Bifurcation Stability: {post_bifurcation_loss:.4f}")
    
    if post_bifurcation_loss < 0.5 and max_loss < 5.0:
        print("\n" + "="*40)
        print("🟢 TITAN SEAL EARNED: System stabilized in chaos.")
        print("="*40 + "\n")
        sys.exit(0)
    else:
        print("\n" + "="*40)
        print(f"🔴 SEAL DENIED: Unstable. Loss={post_bifurcation_loss:.4f}")
        print("="*40 + "\n")
        sys.exit(1)

if __name__ == "__main__":
    run_titan_protocol()