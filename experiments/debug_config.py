import sys
import os
import inspect
from dataclasses import fields

# Path Setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import airbornehrs
print(f"Package Path: {airbornehrs.__file__}")

from airbornehrs.core import AdaptiveFrameworkConfig

print("Config Fields:")
for f in fields(AdaptiveFrameworkConfig):
    print(f"  - {f.name}: {f.type}")

print("\nInit Signature:")
print(inspect.signature(AdaptiveFrameworkConfig.__init__))
