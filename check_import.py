
import sys
import os
sys.path.append(os.getcwd())

try:
    from airbornehrs.memory import RelationalGraphMemory
    print("SUCCESS: RelationalGraphMemory imported")
    obj = RelationalGraphMemory()
    print("SUCCESS: RelationalGraphMemory instantiated")
except Exception as e:
    print(f"FAIL: {e}")
    import traceback
    traceback.print_exc()
