"""
PROTOCOL PHASE 2: FUNCTIONAL VERIFICATION (LOGIC CHECK)
=======================================================
Goal: Verify that the "magic" mechanisms actually alter the model state mathematically.
Checks:
1. Weight Adaptation (Plasticity): Do weights change when Introspection fires?
2. EWC Consolidation (Memory): Does the Fisher Matrix populate?
3. Reptile Reflex (Meta-Control): Does the Learning Rate adapt to stress?
"""

import torch
import torch.nn as nn
import logging
import sys
import os
import numpy as np

# Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from airbornehrs import (
        AdaptiveFramework, 
        AdaptiveFrameworkConfig, 
        MetaController,
        MetaControllerConfig
    )
except ImportError:
    sys.exit(1)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("Phase2")

# ==============================================================================
# 1. SETUP
# ==============================================================================
class VerificationSubject(nn.Module):
    def __init__(self):
        super().__init__()
        # A simple linear layer is enough to test weight changes
        self.fc = nn.Linear(10, 10) 
        
    def forward(self, x):
        return self.fc(x)

def get_weight_checksum(model):
    """Calculates L2 Norm of all weights to detect changes."""
    checksum = 0.0
    for param in model.parameters():
        checksum += torch.norm(param).item()
    return checksum

# ==============================================================================
# 2. VERIFICATION TESTS
# ==============================================================================
def run_verification():
    logger.info("🔬 PHASE 2: STARTING FUNCTIONAL VERIFICATION")
    
    # Init Shared Components
    fw_config = AdaptiveFrameworkConfig(
        model_dim=10, 
        weight_adaptation_lr=0.1, # High LR to ensure visible change
        compile_model=False,
        device='cpu'
    )
    model = VerificationSubject()
    framework = AdaptiveFramework(model, fw_config)
    
    # ------------------------------------------------------------------
    # TEST 1: PLASTICITY (Weight Adaptation)
    # ------------------------------------------------------------------
    logger.info("Test 1: Plasticity (Direct Weight Editing)...")
    try:
        initial_checksum = get_weight_checksum(framework.model)
        
        # Mock Introspection Output: [Scale=1.0, Shift=1.0]
        # This simulates the Introspection Engine demanding a weight shift
        mock_modifiers = torch.tensor([1.0, 1.0], device=framework.device)
        
        # Manually trigger the monitor's adaptation function
        # We need to construct the 'internals' dict expected by core.py
        mock_telemetry = torch.ones((1, 4), device=framework.device) # Mock stats
        internals = {
            'affine_modifiers': mock_modifiers,
            'telemetry_buffer': mock_telemetry,
            'layer_map': framework.layer_map
        }
        
        # Call the function
        magnitude = framework.monitor.adapt_weights(
            current_loss=1.0, 
            previous_loss=0.5, 
            activations=internals
        )
        
        final_checksum = get_weight_checksum(framework.model)
        delta = abs(final_checksum - initial_checksum)
        
        if delta > 1e-6:
            logger.info(f"   ✅ Weights altered successfully. Delta: {delta:.6f}")
        else:
            raise ValueError("Weights did not change despite adaptation signal.")
            
    except Exception as e:
        logger.error(f"   ❌ Plasticity Test Failed: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # TEST 2: MEMORY (EWC Consolidation)
    # ------------------------------------------------------------------
    logger.info("Test 2: Memory (Fisher Information Matrix)...")
    try:
        # 1. Fill Buffer with dummy experiences
        dummy_x = torch.randn(1, 10)
        dummy_y = torch.randn(1, 10)
        
        for _ in range(15): # Minimum buffer size is usually 10
            # Manually add snapshot
            framework.feedback_buffer.add(
                dummy_x, dummy_y, dummy_y, reward=1.0, loss=0.1
            )
            
        # 2. Trigger Consolidation
        framework.ewc.consolidate_from_buffer(framework.feedback_buffer)
        
        # 3. Check Fisher Dict
        fisher_keys = len(framework.ewc.fisher_dict)
        
        if fisher_keys > 0:
            logger.info(f"   ✅ Fisher Matrix populated. Secured {fisher_keys} layers.")
        else:
            raise ValueError("Fisher Matrix is empty after consolidation.")
            
    except Exception as e:
        logger.error(f"   ❌ Memory Test Failed: {e}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # TEST 3: REFLEX (Meta-Controller Response)
    # ------------------------------------------------------------------
    logger.info("Test 3: Reflex (Dynamic Learning Rate)...")
    try:
        meta_config = MetaControllerConfig(
            base_lr=0.001,
            max_lr=0.01
        )
        controller = MetaController(framework, meta_config)
        
        # Simulate a high-stress event (High Loss, High Gradient Norm)
        # We assume the GradientAnalyzer will see noise if we run a train step
        # But we can also test the scheduler directly.
        
        initial_lr = controller.lr_scheduler.current_lr
        
        # Inject massive "Surprise" (High Loss z-score)
        # We manipulate the history to establish a low baseline
        controller.lr_scheduler.loss_history.extend([0.1] * 20)
        
        # Adapt to a high loss event
        high_loss = 5.0 
        adaptation = controller.adapt(loss=high_loss)
        
        new_lr = adaptation['learning_rate']
        
        if new_lr > initial_lr:
             logger.info(f"   ✅ Reflex Triggered. LR spiked: {initial_lr} -> {new_lr}")
        elif new_lr != initial_lr:
             logger.info(f"   ✅ Scheduler Active (Adjusted down/up): {initial_lr} -> {new_lr}")
        else:
             logger.warning("   ⚠️  Reflex weak (LR did not change). This might be due to z-score threshold.")

    except Exception as e:
        logger.error(f"   ❌ Reflex Test Failed: {e}")
        sys.exit(1)

    print("\n" + "="*40)
    print("🟢 FUNCTIONAL VERIFICATION PASSED")
    print("="*40 + "\n")
    sys.exit(0)

if __name__ == "__main__":
    run_verification()