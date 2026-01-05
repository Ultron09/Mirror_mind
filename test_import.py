
import sys
import os
sys.path.insert(0, os.getcwd())
print("DEBUG: Starting import test", flush=True)

try:
    from airbornehrs.core import AdaptiveFramework
    print("DEBUG: Import successful", flush=True)
except Exception as e:
    print(f"DEBUG: Import failed: {e}", flush=True)
    sys.exit(1)

try:
    print("DEBUG: Instantiating AdaptiveFramework", flush=True)
    import torch.nn as nn
    model = nn.Linear(10, 2)
    agent = AdaptiveFramework(model)
    print("DEBUG: Instantiation successful", flush=True)
except Exception as e:
    print(f"DEBUG: Instantiation failed: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
