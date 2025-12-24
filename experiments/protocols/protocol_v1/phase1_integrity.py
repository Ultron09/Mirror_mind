"""
PROTOCOL PHASE 1: SYSTEM INTEGRITY (SMOKE TEST) - WITH REPORTING
================================================================
Goal: Verify that the MirrorMind stack installs, runs, plots, and reports.
Checks:
1. Module Import
2. Component Instantiation
3. Basic Data Flow (Forward/Backward)
4. NaN Detection
5. Automated Visualization & Reporting
"""

import torch
import torch.nn as nn
import logging
import sys
import os
import platform
import datetime
import matplotlib.pyplot as plt

# Ensure we can import the package
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
except ImportError as e:
    print(f"❌ CRITICAL: Import failed. {e}")
    sys.exit(1)

# Configure Clean Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("Phase1")

# ==============================================================================
# HELPER: Visualization & Reporting
# ==============================================================================
def generate_artifacts(losses, configs, status):
    """Generates PNG plot and MD report for research documentation."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # --- 1. Generate Visualization (PNG) ---
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(losses, marker='o', linestyle='-', color='#2c3e50', label='Smoke Test Loss')
        plt.title(f"MirrorMind Integrity Check: Data Flow Verification\n{timestamp}")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig("phase1_loss_plot.png", dpi=300)
        plt.close()
        logger.info("   ✅ Visualization saved: phase1_loss_plot.png")
    except Exception as e:
        logger.error(f"   ⚠️ Visualization failed: {e}")

    # --- 2. Generate Research Report (Markdown) ---
    report_content = f"""# MirrorMind Protocol: Phase 1 Integrity Report
**Date:** {timestamp}
**Status:** {status}

## 1. System Environment
* **OS:** {platform.system()} {platform.release()}
* **Python:** {sys.version.split()[0]}
* **PyTorch:** {torch.__version__}
* **Device:** {configs['device']}

## 2. Configuration
* **Model Dim:** {configs['model_dim']}
* **Meta-Learning Strategy:** Reptile (Active: {configs['use_reptile']})
* **Inference Mode:** Online

## 3. Experimental Output
* **Steps Executed:** {len(losses)}
* **Final Loss:** {losses[-1] if losses else 'N/A'}
* **NaN Detected:** No

## 4. Conclusion
The system successfully instantiated the AdaptiveFramework and MetaController. Forward and backward passes were executed without memory errors or numerical instability.
"""
    
    with open("PHASE1_REPORT.md", "w") as f:
        f.write(report_content)
    logger.info("   ✅ Research Report generated: PHASE1_REPORT.md")

# ==============================================================================
# 1. DUMMY MODEL (The "Subject")
# ==============================================================================
class SimpleSubject(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
        
    def forward(self, x):
        return self.net(x)

# ==============================================================================
# 2. INTEGRITY CHECK
# ==============================================================================
def run_integrity_check():
    logger.info("🚀 PHASE 1: STARTING INTEGRITY CHECK")
    
    loss_history = []
    config_snapshot = {}
    
    # A. Instantiation
    try:
        logger.info("Step 1: Instantiating Components...")
        
        fw_config = AdaptiveFrameworkConfig(
            model_dim=32,
            num_layers=1, 
            learning_rate=0.01,
            compile_model=False,
            device='cpu'
        )
        
        base_model = SimpleSubject()
        framework = AdaptiveFramework(base_model, fw_config)
        
        meta_config = MetaControllerConfig(use_reptile=True)
        controller = MetaController(framework, meta_config)
        
        adapter = ProductionAdapter(framework, inference_mode=InferenceMode.ONLINE)
        
        # Save configs for report
        config_snapshot = {
            'model_dim': 32,
            'device': 'cpu',
            'use_reptile': True
        }
        
        logger.info("   ✅ Components instantiated successfully.")
        
    except Exception as e:
        logger.error(f"   ❌ Instantiation failed: {e}")
        sys.exit(1)

    # B. Execution Loop (Smoke Test)
    try:
        logger.info("Step 2: Executing Training Loop (5 Steps)...")
        
        dummy_input = torch.randn(5, 10) 
        dummy_target = torch.randn(5, 10)
        
        for i in range(5):
            output = adapter.predict(dummy_input, update=True, target=dummy_target)
            
            if torch.isnan(output).any():
                raise ValueError("NaN detected in output!")
                
            metrics = adapter.get_metrics()
            loss = metrics.get('avg_recent_loss', metrics.get('loss', 0.0))
            
            # Record loss for visualization
            if isinstance(loss, torch.Tensor):
                loss_history.append(loss.item())
            else:
                loss_history.append(loss)
            
            if i == 0: logger.info(f"   First Step Loss: {loss:.4f}")
            
        logger.info("   ✅ Training loop completed without crash.")
        
    except Exception as e:
        logger.error(f"   ❌ Execution failed: {e}")
        sys.exit(1)

    # C. State Persistence Check
    try:
        logger.info("Step 3: Checking Checkpoint System...")
        adapter.save_checkpoint("smoke_test.pt")
        if os.path.exists("smoke_test.pt"):
            logger.info("   ✅ Checkpoint created.")
            os.remove("smoke_test.pt")
        else:
            raise FileNotFoundError("Checkpoint file not found.")
            
    except Exception as e:
        logger.error(f"   ❌ Persistence failed: {e}")
        sys.exit(1)

    # D. Artifact Generation (New)
    try:
        logger.info("Step 4: Generating Research Artifacts...")
        generate_artifacts(loss_history, config_snapshot, status="PASSED")
    except Exception as e:
        logger.warning(f"   ⚠️ Artifact generation warning: {e}")

    print("\n" + "="*40)
    print("🟢 SYSTEM STABLE: All integrity checks passed.")
    print("   -> Plot saved to: phase1_loss_plot.png")
    print("   -> Report saved to: PHASE1_REPORT.md")
    print("="*40 + "\n")
    sys.exit(0)

if __name__ == "__main__":
    run_integrity_check()