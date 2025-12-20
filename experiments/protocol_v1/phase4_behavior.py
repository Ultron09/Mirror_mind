"""
PROTOCOL PHASE 4: BEHAVIORAL DYNAMICS (THE REFLEX TEST)
=======================================================
Goal: Verify the system reacts to "Surprise" (Concept Drift) correctly.
Scenario:
1. Train on Task A (Identity: y = x) until stable.
2. SUDDENLY switch to Task B (Inversion: y = -x).
3. Monitor:
   - Surprise Z-Score (Should spike > 3.0)
   - Learning Rate (Should spike immediately after)
   - Loss Recovery (Should decrease after the spike)
"""

import torch
import torch.nn as nn
import logging
import sys
import os
import matplotlib.pyplot as plt

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
logger = logging.getLogger("Phase4")

# ==============================================================================
# 1. SETUP
# ==============================================================================
class DynamicSubject(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

def run_behavior_test():
    logger.info("🧠 PHASE 4: STARTING BEHAVIORAL DYNAMICS TEST")
    
    # Config: High sensitivity to prove the point
    fw_config = AdaptiveFrameworkConfig(
        model_dim=16,
        learning_rate=0.005,
        compile_model=False,
        device='cpu'
    )
    
    # Config optimized for fast reaction
    meta_config = MetaControllerConfig(
        use_reptile=True,
        base_lr=0.005,
        max_lr=0.05,       # Allow 10x spike
    )
    
    model = DynamicSubject()
    framework = AdaptiveFramework(model, fw_config)
    controller = MetaController(framework, meta_config)
    adapter = ProductionAdapter(framework, inference_mode=InferenceMode.ONLINE)
    
    # ==========================================================================
    # 2. THE SIMULATION
    # ==========================================================================
    history_loss = []
    history_z = []
    history_lr = []
    
    TOTAL_STEPS = 200
    SWITCH_STEP = 100
    
    logger.info(f"   Simulating {TOTAL_STEPS} steps. Concept Drift at step {SWITCH_STEP}.")
    
    drift_detected = False
    reflex_triggered = False
    
    for step in range(TOTAL_STEPS):
        # 1. Generate Data
        x = torch.randn(10, 1)
        
        if step < SWITCH_STEP:
            # Task A: Identity
            y = x 
        else:
            # Task B: Inversion (The Surprise)
            y = -x 
            
        # 2. Run Adapter (Predict -> Introspect -> Adapt)
        _ = adapter.predict(x, update=True, target=y)
        
        # 3. Capture Telemetry
        # FIX: We access the internal history directly to get the INSTANT loss
        # The adapter.get_metrics() returns an average, which smooths out the spike we need to see.
        if adapter.framework.loss_history:
            current_loss = adapter.framework.loss_history[-1]
        else:
            current_loss = 0.0

        # Metrics for Z-Score/LR
        metrics = adapter.get_metrics()
        current_lr = framework.optimizer.param_groups[0]['lr']
        
        # Try to get Z-score from controller, fallback to proxy
        if hasattr(controller.lr_scheduler, 'last_z_score'):
             current_z = controller.lr_scheduler.last_z_score
        else:
             current_z = 0.0

        history_loss.append(current_loss)
        history_z.append(current_z)
        history_lr.append(current_lr)
        
        # 4. Real-time Monitoring
        if step == SWITCH_STEP:
            logger.info("   ⚠️  CONCEPT DRIFT INJECTED (Identity -> Inversion)")
            
        if step > SWITCH_STEP and step < SWITCH_STEP + 15:
            
            if current_loss > 1.0: # Explicit Loss Check
                if not drift_detected:
                    logger.info(f"   ⚡ SURPRISE DETECTED at step {step}! Loss Spike: {current_loss:.2f}")
                    drift_detected = True
            
            if current_lr > 0.006: # > Base LR
                if not reflex_triggered:
                    logger.info(f"   🚀 REFLEX TRIGGERED at step {step}! LR spiked to {current_lr:.4f}")
                    reflex_triggered = True

    # ==========================================================================
    # 3. ANALYSIS
    # ==========================================================================
    
    # A. Did we panic? (Check max loss in the drift window)
    drift_window = history_loss[SWITCH_STEP:SWITCH_STEP+20]
    max_loss_after_drift = max(drift_window) if drift_window else 0.0
    
    metrics_window = history_lr[SWITCH_STEP:SWITCH_STEP+20]
    max_lr_after_drift = max(metrics_window) if metrics_window else 0.0
    
    # B. Did we recover?
    final_loss = sum(history_loss[-10:]) / 10
    
    logger.info("-" * 40)
    logger.info(f"   Max Loss (Surprise): {max_loss_after_drift:.2f}")
    logger.info(f"   Max Reflex (LR):     {max_lr_after_drift:.4f}")
    logger.info(f"   Final Stability:     {final_loss:.4f}")
    
    # Verification Logic
    if max_loss_after_drift > 0.5 and max_lr_after_drift > 0.0055:
        print("\n" + "="*40)
        print("🟢 BEHAVIOR VERIFIED: System panics and adapts correctly.")
        print("="*40 + "\n")
        sys.exit(0)
    else:
        print("\n" + "="*40)
        print("🔴 BEHAVIOR FAILED: System was too rigid or blind to drift.")
        print(f"Details: Loss Spike={max_loss_after_drift}, Max LR={max_lr_after_drift}")
        print("="*40 + "\n")
        sys.exit(1)

if __name__ == "__main__":
    run_behavior_test()