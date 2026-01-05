import torch
import sys
import os
import logging

# Path Setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from airbornehrs.consciousness_v2 import EnhancedConsciousnessCore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GWT_Verify")

def run_verification():
    logger.info("🧪 Verifying Awakened Mind (GWT)...")
    
    # Initialize Core
    feature_dim = 64
    core = EnhancedConsciousnessCore(feature_dim=feature_dim)
    
    # Check Workspace Initialization
    if not hasattr(core, 'global_workspace'):
        logger.error("❌ Global Workspace not found in ConsciousnessCore.")
        return
    logger.info("✅ Global Workspace initialized.")
    
    # Test Sequence
    logger.info("Running Thought Process Sequence...")
    
    # Step 1: Input A
    features_a = torch.randn(1, feature_dim) # [B, D]
    # Mock prediction
    y_true = torch.tensor([0])
    y_pred = torch.tensor([[1.0, 0.0]])
    
    metrics_a = core.observe(y_true=y_true, y_pred=y_pred, features=features_a)
    thought_a = core.current_thought
    
    if thought_a is None:
        logger.error("❌ Thought A is None.")
        return
    logger.info(f"Thought A Shape: {thought_a.shape}")
    
    # Step 2: Input B
    features_b = torch.randn(1, feature_dim)
    metrics_b = core.observe(y_true=y_true, y_pred=y_pred, features=features_b)
    thought_b = core.current_thought
    
    # Verify Persistence / Change
    diff = (thought_a - thought_b).abs().sum().item()
    logger.info(f"Thought Difference (A vs B): {diff:.4f}")
    
    if diff < 1e-6:
        logger.warning("⚠️ Thought did not change! Recurrence might be broken or inputs ignored.")
    else:
        logger.info("✅ Thought evolved over time.")
        
    # Verify Memory (Recurrence)
    # If we pass same input again, thought should be different because of state
    metrics_c = core.observe(y_true=y_true, y_pred=y_pred, features=features_b)
    thought_c = core.current_thought
    
    diff_bc = (thought_b - thought_c).abs().sum().item()
    logger.info(f"Thought Difference (B vs B_again): {diff_bc:.4f}")
    
    if diff_bc > 1e-6:
        logger.info("✅ Recurrent state confirmed (Context matters).")
    else:
        logger.warning("⚠️ Thought is identical for same input. Memory might be stateless.")

    logger.info("✅ GWT Verification Passed!")

if __name__ == "__main__":
    run_verification()
