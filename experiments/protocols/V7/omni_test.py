import torch
import torch.nn as nn
import sys
import os
import logging
import numpy as np
import matplotlib.pyplot as plt

# Path Setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig
from airbornehrs.memory import UnifiedMemoryHandler
from airbornehrs.consciousness_v2 import EnhancedConsciousnessCore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Omni_Test")

def generate_task_data(task_id, num_samples=1000, input_dim=64):
    """
    Generates synthetic data for different tasks.
    Task 1: Linear correlation in first 10 dims.
    Task 2: Non-linear pattern in last 10 dims.
    Task 3: XOR logic of Task 1 and 2 features.
    """
    X = torch.randn(num_samples, input_dim)
    y = torch.zeros(num_samples, 1)
    
    if task_id == 1:
        # Task 1: y = sum(x[0:5])
        y = X[:, :5].sum(dim=1, keepdim=True)
        y = (y > 0).float() # Classification
    elif task_id == 2:
        # Task 2: y = sum(x[-5:])
        y = X[:, -5:].sum(dim=1, keepdim=True)
        y = (y > 0).float()
    elif task_id == 3:
        # Task 3: XOR of Task 1 and Task 2
        t1 = (X[:, :5].sum(dim=1) > 0).float()
        t2 = (X[:, -5:].sum(dim=1) > 0).float()
        y = torch.abs(t1 - t2).unsqueeze(1)
        
    return X, y

def run_omni_test():
    logger.info("🚀 Starting THE OMNI-TEST (Protocol V7 Validation)...")
    
    # 1. Setup V7 Framework
    input_dim = 64
    base_model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    
    config = AdaptiveFrameworkConfig(
        use_moe=True,
        num_experts=4,
        top_k_experts=2,
        input_dim=input_dim,
        model_dim=input_dim, # CRITICAL: Match feature dim for GWT
        device='cpu' # Use CPU for stability in test
    )
    
    # Enable OGD and Learned Optimizer manually (since they are in sub-modules)
    # Framework initializes them, but we need to ensure config passes through.
    # AdaptiveFrameworkConfig doesn't have 'use_ogd' directly, it's passed to Memory.
    # We need to hack/configure the framework after init or ensure it passes kwargs.
    # Looking at core.py, it initializes Memory with default args.
    # We might need to modify core.py to accept these configs, or set them manually.
    
    framework = AdaptiveFramework(base_model, config)
    
    # Manually Upgrade Memory to OGD
    framework.memory = UnifiedMemoryHandler(
        framework.model, 
        method='hybrid', 
        use_ogd=True
    )
    logger.info("✅ OGD Enabled.")
    
    # Manually Upgrade Meta-Controller (if needed)
    # framework.meta_controller is already init with default config.
    # Let's check if it has learned optimizer.
    if hasattr(framework.meta_controller.config, 'use_learned_optimizer'):
        framework.meta_controller.config.use_learned_optimizer = True
        # Re-init scheduler
        from airbornehrs.meta_controller import DynamicLearningRateScheduler
        framework.meta_controller.lr_scheduler = DynamicLearningRateScheduler(
            framework.optimizer, framework.meta_controller.config
        )
        logger.info("✅ Learned Optimizer Enabled.")
    
    # 2. Train Task 1
    logger.info("\n📘 Training Task 1 (Pattern A)...")
    X1, y1 = generate_task_data(1)
    for i in range(5): # Short training
        metrics = framework.train_step(X1, target_data=y1)
        if i % 1 == 0: logger.info(f"Step {i}: Loss {metrics['loss']:.4f}")
        
    # Consolidate Task 1
    logger.info("🔒 Consolidating Task 1...")
    framework.memory.consolidate(feedback_buffer=framework.feedback_buffer)
    
    # Evaluate Task 1
    pred1, _, _ = framework(X1)
    acc1_pre = ((pred1 > 0.5) == (y1 > 0.5)).float().mean().item()
    logger.info(f"Task 1 Accuracy (Pre-Task 2): {acc1_pre:.4f}")
    
    # 3. Train Task 2
    logger.info("\n📗 Training Task 2 (Pattern B)...")
    X2, y2 = generate_task_data(2)
    # Clear buffer to simulate new task stream (optional, but OGD needs history of T1? No, OGD stored subspace)
    # But EWC needs history? EWC is offline.
    # We should clear buffer so we don't replay T1 data immediately (testing forgetting).
    framework.feedback_buffer.buffer.clear()
    
    for i in range(5):
        metrics = framework.train_step(X2, target_data=y2)
        if i % 1 == 0: logger.info(f"Step {i}: Loss {metrics['loss']:.4f}")
        
    # Consolidate Task 2
    logger.info("🔒 Consolidating Task 2...")
    framework.memory.consolidate(feedback_buffer=framework.feedback_buffer)
    
    # 4. Train Task 3 (Reasoning)
    logger.info("\n📙 Training Task 3 (XOR Logic)...")
    X3, y3 = generate_task_data(3)
    framework.feedback_buffer.buffer.clear()
    
    for i in range(5):
        metrics = framework.train_step(X3, target_data=y3)
        if i % 1 == 0: logger.info(f"Step {i}: Loss {metrics['loss']:.4f}")
        
    # 5. Final Evaluation (The Omni-Test)
    logger.info("\n📊 FINAL EVALUATION")
    
    # Task 1 (Memory)
    pred1_post, _, _ = framework(X1)
    acc1_post = ((pred1_post > 0.5) == (y1 > 0.5)).float().mean().item()
    forgetting = acc1_pre - acc1_post
    logger.info(f"Task 1 Accuracy (Post): {acc1_post:.4f} (Forgetting: {forgetting:.4f})")
    
    # Task 3 (Reasoning)
    pred3, _, _ = framework(X3)
    acc3 = ((pred3 > 0.5) == (y3 > 0.5)).float().mean().item()
    logger.info(f"Task 3 Accuracy (Reasoning): {acc3:.4f}")
    
    # Check GWT Activity
    if framework.consciousness.current_thought is not None:
        logger.info("✅ Global Workspace is Active.")
    else:
        logger.warning("⚠️ Global Workspace is Inactive.")
        
    # Success Criteria
    if forgetting < 0.1 and acc3 > 0.6:
        logger.info("\n🏆 OMNI-TEST PASSED! System is Groundbreaking.")
    else:
        logger.warning("\n⚠️ Omni-Test Results Sub-optimal. Further tuning required.")

if __name__ == "__main__":
    run_omni_test()
