
"""
Extended monkey-patch runner for mirrormind_agi.py

Adds legality check inside Environment.step():
- If an action decodes to an illegal move, it penalizes and terminates instead of crashing python-chess.
"""

import os, sys, torch
import mirrormind_agi as base

ACTION_SPACE_SIZE = getattr(base, "ACTION_SPACE_SIZE", 64*64)

# Resolve CLS index
try:
    CLS_INDEX = base.SpecialTokens.CLS
except Exception:
    CLS_INDEX = 13

# -------- reuse prior patches if available --------
# (we import monkey_patch first version if present)
try:
    import mirrormind_agi_monkey_patch as prev_patch
except Exception:
    pass

# -------- Patch Environment.step to guard illegal moves --------
_env_cls = None
for cand in ("AGIChessEnvironment", "ChessEnvironment"):
    if hasattr(base, cand) and isinstance(getattr(base, cand), type):
        _env_cls = getattr(base, cand)
        break

if _env_cls is not None and hasattr(_env_cls, "step"):
    _ENV_orig_step = _env_cls.step

    def _ENV_patched_step(self, action, reasoning_trace=None, consciousness_state=None):
        try:
            move = self.decode_action(action)
        except Exception as e:
            print(f"⚠️ Could not decode action {action}: {e}")
            return self.get_state(), -10, True, {"illegal_move": True, "decode_error": True}

        # Check legality against current board
        if move not in self.board.legal_moves:
            print(f"⚠️ Illegal move attempted: {move} at {self.board.fen()}")
            reward = -10
            done = True
            info = {"illegal_move": True}
            return self.get_state(), reward, done, info

        # Otherwise call original step logic safely
        return _ENV_orig_step(self, action, reasoning_trace, consciousness_state)

    _env_cls.step = _ENV_patched_step

# ---------- Run original CLI if present ----------
if __name__ == "__main__":
    if hasattr(base, "main") and callable(getattr(base, "main")):
        sys.exit(base.main())
    print("Patched mirrormind_agi imported. No `main()` found to run.")
