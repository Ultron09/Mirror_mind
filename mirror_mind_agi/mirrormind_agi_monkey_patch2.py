import os
import re

TARGET_FILE = "mirrormind_agi.py"
BACKUP_FILE = TARGET_FILE + ".bak"

# Backup original file once
if not os.path.exists(BACKUP_FILE):
    os.rename(TARGET_FILE, BACKUP_FILE)
    print(f"📦 Backup saved as {BACKUP_FILE}")

with open(BACKUP_FILE, "r", encoding="utf-8") as f:
    code = f.read()

# ===================== CPU LOCK HEADER =====================
header = r"""
# ===================== PATCH 7 START =====================
import os, torch

# Kill CUDA globally
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.cuda.is_available = lambda : False
torch.device = lambda *args, **kwargs: torch.device("cpu")

print("🖥️ Running on CPU only (Patch 7 applied)")

# Force all new tensors to CPU
_old_tensor = torch.tensor
def _cpu_tensor(*args, **kwargs):
    return _old_tensor(*args, **kwargs).to("cpu")
torch.tensor = _cpu_tensor
# ===================== PATCH 7 END =====================
"""

# Replace explicit "cuda" mentions with "cpu"
code = re.sub(r'["\']cuda["\']', '"cpu"', code)

# Prepend header
patched_code = header + "\n" + code

with open(TARGET_FILE, "w", encoding="utf-8") as f:
    f.write(patched_code)

print(f"✅ Patch 7 applied → {TARGET_FILE} is now hard-locked to CPU")
