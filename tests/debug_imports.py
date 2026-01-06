import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("Checking imports...")

try:
    from airbornehrs.core import AdaptiveFramework
    print("✓ AdaptiveFramework found in core")
except ImportError as e:
    print(f"✗ AdaptiveFramework NOT found: {e}")

try:
    from airbornehrs.consciousness_v2 import EnhancedConsciousnessCore
    print("✓ EnhancedConsciousnessCore found in consciousness_v2")
except ImportError as e:
    print(f"✗ EnhancedConsciousnessCore NOT found: {e}")

try:
    from airbornehrs.memory import UnifiedMemoryHandler
    print("✓ UnifiedMemoryHandler found in memory")
except ImportError as e:
    print(f"✗ UnifiedMemoryHandler NOT found: {e}")
