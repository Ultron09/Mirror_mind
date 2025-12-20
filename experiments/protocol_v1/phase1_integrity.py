"""
PROTOCOL PHASE 1: SYSTEM INTEGRITY (SMOKE TEST)
===============================================
Goal: Verify that the MirrorMind stack installs and runs without crashing.
Checks:
1. Module Import
2. Component Instantiation (Framework, Controller, Adapter)
3. Basic Data Flow (Forward/Backward)
4. NaN Detection
"""

import torch
import torch.nn as nn
import logging
import sys
import os

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
    
    # A. Instantiation
    try:
        logger.info("Step 1: Instantiating Components...")
        
        fw_config = AdaptiveFrameworkConfig(
            model_dim=32,
            num_layers=1, # Minimal
            learning_rate=0.01,
            compile_model=False, # Disable compile for smoke test
            device='cpu'
        )
        
        base_model = SimpleSubject()
        framework = AdaptiveFramework(base_model, fw_config)
        
        meta_config = MetaControllerConfig(use_reptile=True)
        controller = MetaController(framework, meta_config)
        
        adapter = ProductionAdapter(framework, inference_mode=InferenceMode.ONLINE)
        
        logger.info("   ✅ Components instantiated successfully.")
        
    except Exception as e:
        logger.error(f"   ❌ Instantiation failed: {e}")
        sys.exit(1)

    # B. Execution Loop (Smoke Test)
    try:
        logger.info("Step 2: Executing Training Loop (5 Steps)...")
        
        dummy_input = torch.randn(5, 10) # Batch 5, Dim 10
        dummy_target = torch.randn(5, 10)
        
        for i in range(5):
            # Test complete flow: Predict -> Introspect -> Update -> Reptile
            output = adapter.predict(dummy_input, update=True, target=dummy_target)
            
            # Check for NaNs
            if torch.isnan(output).any():
                raise ValueError("NaN detected in output!")
                
            metrics = adapter.get_metrics()
            loss = metrics.get('loss', 0.0)
            
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

    print("\n" + "="*40)
    print("🟢 SYSTEM STABLE: All integrity checks passed.")
    print("="*40 + "\n")
    sys.exit(0)

if __name__ == "__main__":
    run_integrity_check()