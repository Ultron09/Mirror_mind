
import sys
import os
# Add path to find benchmark_continual BEFORE importing from it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig
from benchmark_continual import ResNetLike  # Import actual model class

# Force verification on correct device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running Integration Check on: {device}")

def run_integration_prover():
    # 1. SETUP FULL CONFIG (Matching Benchmark)
    cfg = AdaptiveFrameworkConfig(device=device)
    cfg.memory_type = 'hybrid'
    cfg.ewc_lambda = 2000.0
    cfg.dream_interval = 2
    cfg.dream_batch_size = 64
    cfg.use_amp = True # Verify AMP
    cfg.compile_model = False # Skip compilation for quick test (compilation takes time)
    cfg.enable_consciousness = True
    
    # 2. INSTANTIATE REAL MODEL
    print(">>> Instantiating ResNet-18 (Airborne)...")
    try:
        model = AdaptiveFramework(ResNetLike(), cfg, device=device)
    except Exception as e:
        print(f"FATAL: Model Init Failed: {e}")
        return False
        
    # 3. DUMMY DATA (CIFAR Size)
    x = torch.randn(64, 3, 32, 32).to(device)
    y = torch.randint(0, 100, (64,)).to(device)
    
    # 4. STEP 1: TRAIN (Phase A)
    print(">>> Step 1: Training Task A (AMP + Backprop)...")
    try:
        metrics = model.train_step(x, target_data=y)
        print(f"    Success. Loss: {metrics['loss']:.4f}")
    except Exception as e:
        print(f"FATAL: Training Step Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 5. STEP 2: CONSOLIDATE (EWC)
    # Need meaningful gradients for Fisher, so run a few backward passes first
    metrics = model.train_step(x, target_data=y)
    
    print(">>> Step 2: Consolidating Memory (Fisher Matrix)...")
    try:
        model.memory.consolidate(feedback_buffer=model.prioritized_buffer)
        num_params = len(model.memory.fisher_dict)
        print(f"    Success. Fisher Matrix calculated for {num_params} tensors.")
        if num_params == 0:
            print("WARNING: Fisher Dict is empty!")
            return False
    except Exception as e:
        print(f"FATAL: Consolidation Failed: {e}")
        return False

    # 6. STEP 3: TRAIN (Phase B + Replay)
    print(">>> Step 3: Training Task B (with Task A Replay)...")
    try:
        # Should trigger replay if dream_interval=2 (we are at step 2 now)
        # Note: prioritized_buffer needs data. train_step adds it automatically.
        metrics = model.train_step(x, target_data=y)
        print(f"    Success. Loss: {metrics['loss']:.4f}")
    except Exception as e:
        print(f"FATAL: Replay/Task B Step Failed: {e}")
        traceback.print_exc()
        return False

    print("\n[OK] INTEGRATION VERIFIED. NO CRASHES.")
    return True

if __name__ == "__main__":
    success = run_integration_prover()
    if not success:
        sys.exit(1)
