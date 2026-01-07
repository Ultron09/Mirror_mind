
import torch
import torch.nn as nn
from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig
import sys
import os

def test_demo_configuration():
    print("[TEST] Initializing Demo Configuration...")
    
    # mimic InteractiveDemo.run_quick_demo logic
    try:
        config = AdaptiveFrameworkConfig.production()
        
        # Override for interactive demo speed
        config.warmup_steps = 2
        config.novelty_threshold = 0.5
        config.panic_threshold = 0.8
        config.enable_consciousness = True
        config.enable_world_model = True 
        
        # CPU/CUDA Check
        if not torch.cuda.is_available():
            config.device = 'cpu'
            print("[TEST] CPU Mode confirmed (CUDA unavailable)")
        else:
            print(f"[TEST] CUDA Mode confirmed: {torch.cuda.get_device_name(0)}")

        model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        framework = AdaptiveFramework(model, config)
        print("[TEST] Framework initialized successfully.")
        
        # Run a step
        x = torch.randn(4, 10).to(framework.device)
        y = torch.randn(4, 1).to(framework.device)
        
        print("[TEST] Running train_step...")
        metrics = framework.train_step(x, target_data=y)
        print(f"[TEST] Step complete. Metrics keys: {list(metrics.keys())}")
        
        # Verify Consciousness Metrics
        if framework.consciousness:
            if hasattr(framework.consciousness, 'last_metrics') and framework.consciousness.last_metrics:
                print("[TEST] SUCCESS: framework.consciousness.last_metrics is populated.")
                print(f"[TEST] Last Metrics: {framework.consciousness.last_metrics}")
            else:
                print("[TEST] FAILURE: framework.consciousness.last_metrics is MISSING or EMPTY.")
                sys.exit(1)
        else:
            print("[TEST] FAILURE: Consciousness module not enabled.")
            sys.exit(1)

    except Exception as e:
        print(f"[TEST] CRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_demo_configuration()
