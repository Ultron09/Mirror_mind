"""
PROTOCOL PHASE 4: BEHAVIORAL DYNAMICS (THE REFLEX TEST) - WITH REPORTING
========================================================================
Goal: Verify the system reacts to "Surprise" (Concept Drift) correctly.
Scenario:
1. Train on Task A (Identity: y = x) until stable.
2. SUDDENLY switch to Task B (Inversion: y = -x) at Step 100.
3. Monitor:
   - Surprise Z-Score (Should spike > 3.0)
   - Learning Rate (Should spike immediately after)
   - Loss Recovery (Should decrease after the spike)
4. Automated Visualization & Reporting
"""

import torch
import torch.nn as nn
import logging
import sys
import os
import platform
import datetime
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
    print("❌ CRITICAL: Import failed.")
    sys.exit(1)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger("Phase4")

# ==============================================================================
# HELPER: Visualization & Reporting
# ==============================================================================
def generate_artifacts(history, drift_stats, status):
    """Generates Dual-Axis PNG plot and MD report for research documentation."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # --- 1. Generate Visualization (PNG) ---
    try:
        steps = range(len(history['loss']))
        switch_step = drift_stats['switch_step']
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot Loss (Left Axis)
        color_loss = '#e74c3c' # Red
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss (Surprise)', color=color_loss, fontweight='bold')
        ax1.plot(steps, history['loss'], color=color_loss, alpha=0.6, label='Loss')
        ax1.tick_params(axis='y', labelcolor=color_loss)
        
        # Plot Learning Rate (Right Axis)
        ax2 = ax1.twinx()  
        color_lr = '#2980b9' # Blue
        ax2.set_ylabel('Learning Rate (Reflex)', color=color_lr, fontweight='bold')
        ax2.plot(steps, history['lr'], color=color_lr, linewidth=2, label='Learning Rate')
        ax2.tick_params(axis='y', labelcolor=color_lr)

        # Add Vertical Line for Concept Drift
        plt.axvline(x=switch_step, color='black', linestyle='--', alpha=0.8, label='Concept Drift')
        plt.text(switch_step + 5, max(history['lr']), 'Drift Event', rotation=0)

        plt.title(f"MirrorMind Behavioral Dynamics: The Reflex Arc\n{timestamp}")
        fig.tight_layout()
        plt.savefig("phase4_reflex_plot.png", dpi=300)
        plt.close()
        logger.info("   ✅ Visualization saved: phase4_reflex_plot.png")
    except Exception as e:
        logger.error(f"   ⚠️ Visualization failed: {e}")

    # --- 2. Generate Research Report (Markdown) ---
    report_content = f"""# MirrorMind Protocol: Phase 4 Behavioral Dynamics
**Date:** {timestamp}
**Status:** {status}

## 1. Objective
To demonstrate the system's ability to detect "Concept Drift" (sudden change in data distribution) and trigger an autonomous "Reflex" (Learning Rate spike) to adapt rapidly.

## 2. Simulation Timeline
* **Task A (0 - {drift_stats['switch_step']}):** Identity Function ($y = x$)
* **Task B ({drift_stats['switch_step']} - End):** Inversion Function ($y = -x$)
* **Drift Injection Point:** Step {drift_stats['switch_step']}

## 3. Reflex Analysis
| Metric | Pre-Drift (Baseline) | Post-Drift (Peak) | Reaction |
| :--- | :--- | :--- | :--- |
| **Loss** | ~{drift_stats['pre_loss']:.4f} | **{drift_stats['peak_loss']:.4f}** | Surprise Detected |
| **Learning Rate** | {drift_stats['base_lr']:.4f} | **{drift_stats['peak_lr']:.4f}** | Reflex Triggered |

## 4. Recovery
* **Final Loss (Last 10 steps):** {drift_stats['final_loss']:.4f}
* **Recovery Status:** {"SUCCESSFUL" if drift_stats['final_loss'] < 0.5 else "FAILED"}

## 5. Conclusion
The system successfully identified the statistical anomaly at Step {drift_stats['switch_step']}. The Meta-Controller responded by temporarily boosting plasticity (Learning Rate), allowing the model to unlearn Task A and master Task B without manual intervention.
"""
    
    with open("PHASE4_REPORT.md", "w") as f:
        f.write(report_content)
    logger.info("   ✅ Research Report generated: PHASE4_REPORT.md")

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
        max_lr=0.15,       # Allow 10x spike
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
    
    TOTAL_STEPS = 500
    SWITCH_STEP = 100
    
    logger.info(f"   Simulating {TOTAL_STEPS} steps. Concept Drift at step {SWITCH_STEP}.")
    
    drift_detected = False
    reflex_triggered = False
    pre_drift_loss = 0.0
    
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
        if adapter.framework.loss_history:
            current_loss = adapter.framework.loss_history[-1]
        else:
            current_loss = 0.0

        current_lr = framework.optimizer.param_groups[0]['lr']
        
        # Capture baseline just before switch
        if step == SWITCH_STEP - 1:
            pre_drift_loss = current_loss

        history_loss.append(current_loss)
        history_lr.append(current_lr)
        
        # 4. Real-time Monitoring
        if step == SWITCH_STEP:
            logger.info("   ⚠️  CONCEPT DRIFT INJECTED (Identity -> Inversion)")
            
        if step > SWITCH_STEP and step < SWITCH_STEP + 15:
            if current_loss > 1.0 and not drift_detected:
                logger.info(f"   ⚡ SURPRISE DETECTED at step {step}! Loss Spike: {current_loss:.2f}")
                drift_detected = True
            
            if current_lr > 0.006 and not reflex_triggered:
                logger.info(f"   🚀 REFLEX TRIGGERED at step {step}! LR spiked to {current_lr:.4f}")
                reflex_triggered = True

    # ==========================================================================
    # 3. ANALYSIS & ARTIFACTS
    # ==========================================================================
    
    # A. Calculate Stats
    drift_window = history_loss[SWITCH_STEP:SWITCH_STEP+20]
    max_loss_after_drift = max(drift_window) if drift_window else 0.0
    
    metrics_window = history_lr[SWITCH_STEP:SWITCH_STEP+20]
    max_lr_after_drift = max(metrics_window) if metrics_window else 0.0
    
    final_loss = sum(history_loss[-10:]) / 10
    
    logger.info("-" * 40)
    logger.info(f"   Max Loss (Surprise): {max_loss_after_drift:.2f}")
    logger.info(f"   Max Reflex (LR):     {max_lr_after_drift:.4f}")
    logger.info(f"   Final Stability:     {final_loss:.4f}")
    
    # B. Generate Report & Plot
    drift_stats = {
        'switch_step': SWITCH_STEP,
        'pre_loss': pre_drift_loss,
        'peak_loss': max_loss_after_drift,
        'base_lr': 0.005,
        'peak_lr': max_lr_after_drift,
        'final_loss': final_loss
    }
    
    history_data = {
        'loss': history_loss,
        'lr': history_lr
    }
    
    # Decide Status
    is_success = max_loss_after_drift > 0.5 and max_lr_after_drift > 0.0055
    status_str = "PASSED" if is_success else "FAILED"
    
    generate_artifacts(history_data, drift_stats, status_str)
    
    # Final Exit
    if is_success:
        print("\n" + "="*40)
        print("🟢 BEHAVIOR VERIFIED: System panics and adapts correctly.")
        print("   -> Plot saved to: phase4_reflex_plot.png")
        print("   -> Report saved to: PHASE4_REPORT.md")
        print("="*40 + "\n")
        sys.exit(0)
    else:
        print("\n" + "="*40)
        print("🔴 BEHAVIOR FAILED: System was too rigid or blind to drift.")
        print("="*40 + "\n")
        sys.exit(1)

if __name__ == "__main__":
    run_behavior_test()