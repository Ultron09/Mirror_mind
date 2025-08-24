
"""
Monkey-patch runner for mirrormind_agi.py

This file imports the user's original module and applies safe patches at runtime:
- Expands piece embedding to include CLS (index 13) if needed.
- Fixes chess geometry indexing when CLS is present (no rank/file out-of-range).
- Ensures board state always prepends CLS in get_state().
- Guards replay buffer .append() so illegal/out-of-range actions never enter training.
Then, if the original module exposes main(), it calls it so your CLI keeps working.

Usage:
    python mirrormind_agi_monkey_patch.py
"""

import os
import sys
import types
import torch
import torch.nn as nn

import mirrormind_agi as base

ACTION_SPACE_SIZE = getattr(base, "ACTION_SPACE_SIZE", 64*64)

# Resolve SpecialTokens.CLS if present, else fallback to 13
try:
    CLS_INDEX = base.SpecialTokens.CLS
except Exception:
    CLS_INDEX = 13

# ---------- Patch UniversalEmbedding.__init__ to ensure embedding sizes ----------
if hasattr(base, "UniversalEmbedding"):
    _UE_orig_init = base.UniversalEmbedding.__init__

    def _UE_patched_init(self, *args, **kwargs):
        _UE_orig_init(self, *args, **kwargs)

        # Ensure piece embedding has room for CLS index
        if hasattr(self, "piece_embed") and isinstance(self.piece_embed, nn.Embedding):
            if self.piece_embed.num_embeddings <= CLS_INDEX:
                new_pe = nn.Embedding(CLS_INDEX + 1, self.piece_embed.embedding_dim)
                with torch.no_grad():
                    new_pe.weight.zero_()
                    new_pe.weight[: self.piece_embed.num_embeddings].copy_(self.piece_embed.weight.data)
                self.piece_embed = new_pe

        # Ensure other chess embeddings meet minimum sizes
        def _ensure_size(attr, min_n):
            emb = getattr(self, attr, None)
            if isinstance(emb, nn.Embedding) and emb.num_embeddings < min_n:
                new_e = nn.Embedding(min_n, emb.embedding_dim)
                with torch.no_grad():
                    new_e.weight.zero_()
                    new_e.weight[: emb.num_embeddings].copy_(emb.weight.data)
                setattr(self, attr, new_e)

        _ensure_size("chess_pos_embed", 65)
        _ensure_size("rank_embed", 8)
        _ensure_size("file_embed", 8)
        _ensure_size("diagonal_embed", 15)
        _ensure_size("anti_diagonal_embed", 15)

    base.UniversalEmbedding.__init__ = _UE_patched_init

    # ---------- Patch UniversalEmbedding._embed_chess with safe indexing ----------
    def _UE_embed_chess_safe(self, board_tensor, positions=None):
        """
        board_tensor: LongTensor [B, L], values include:
            0..12 (empty + pieces) and possible CLS_INDEX at position 0 if L==65
        """
        batch_size, seq_len = board_tensor.shape

        # Piece embedding
        piece_emb = self.piece_embed(board_tensor)

        # Positional embeddings
        if positions is None:
            positions = torch.arange(seq_len, device=board_tensor.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.chess_pos_embed(positions.clamp(0, self.chess_pos_embed.num_embeddings - 1))

        # Geometry indexes computed over squares only (exclude CLS if present)
        board_positions = positions[:, 1:] if seq_len == 65 else positions
        offset = 1 if seq_len == 65 else 0
        squares = (board_positions - offset).clamp(0, 63)  # 0..63

        ranks = squares // 8
        files = squares % 8
        diagonals = ranks + files            # 0..14
        anti_diagonals = ranks - files + 7   # 0..14

        # Clamp with debug prints if out-of-range happens
        def _check(name, x, max_n):
            minv = int(x.min().item())
            maxv = int(x.max().item())
            if minv < 0 or maxv >= max_n:
                print(f"🚨 {name} out of range: min={minv} max={maxv} valid=[0,{max_n-1}]  (auto-clamped)")
            return x.clamp(0, max_n-1)

        ranks = _check("rank", ranks, self.rank_embed.num_embeddings)
        files = _check("file", files, self.file_embed.num_embeddings)
        diagonals = _check("diag", diagonals, self.diagonal_embed.num_embeddings)
        anti_diagonals = _check("anti_diag", anti_diagonals, self.anti_diagonal_embed.num_embeddings)

        rank_emb = self.rank_embed(ranks)
        file_emb = self.file_embed(files)
        diag_emb = self.diagonal_embed(diagonals)
        anti_diag_emb = self.anti_diagonal_embed(anti_diagonals)
        geom_emb = torch.cat([rank_emb, file_emb, diag_emb, anti_diag_emb], dim=-1)

        # Prepend a zero geom row for CLS if present
        if seq_len == 65:
            zero_geom = torch.zeros(batch_size, 1, geom_emb.size(-1), device=board_tensor.device)
            geom_emb = torch.cat([zero_geom, geom_emb], dim=1)

        return piece_emb + pos_emb + geom_emb

    base.UniversalEmbedding._embed_chess = _UE_embed_chess_safe

# ---------- Patch Environment.get_state to always prepend CLS ----------
_env_cls = None
for cand in ("AGIChessEnvironment", "ChessEnvironment"):
    if hasattr(base, cand) and isinstance(getattr(base, cand), type):
        _env_cls = getattr(base, cand)
        break

if _env_cls is not None and hasattr(_env_cls, "get_state"):
    _ENV_orig_get_state = _env_cls.get_state

    def _ENV_patched_get_state(self):
        state = _ENV_orig_get_state(self)
        try:
            board = state["board"]
        except Exception:
            return state

        if isinstance(board, torch.Tensor):
            # 1D tensor expected
            if board.dim() == 1 and board.size(0) == 64:
                cls_token = torch.full((1,), CLS_INDEX, dtype=board.dtype, device=board.device)
                state["board"] = torch.cat([cls_token, board], dim=0)
            elif board.dim() == 1 and board.size(0) == 65 and int(board[0].item()) != CLS_INDEX:
                # If it's 65 but doesn't start with CLS, fix it
                cls_token = torch.full((1,), CLS_INDEX, dtype=board.dtype, device=board.device)
                state["board"] = torch.cat([cls_token, board[1:]], dim=0)
        return state

    _env_cls.get_state = _ENV_patched_get_state

# ---------- Guard replay buffer append so illegal actions never enter ----------
if hasattr(base, "AGITrainer") and isinstance(base.AGITrainer, type):
    _Trainer = base.AGITrainer
    _TR_orig_init = _Trainer.__init__

    class GuardedBuffer:
        def __init__(self, inner, action_space_size):
            self._inner = inner
            self._asz = int(action_space_size)

        def append(self, experience):
            try:
                action = experience.get("action", -1)
                legal = experience.get("legal_move", True)
            except Exception:
                action, legal = -1, True

            if isinstance(action, int) and 0 <= action < self._asz and legal:
                return self._inner.append(experience)
            else:
                print(f"⚠️ Skipped illegal/out-of-range action: {action}")
                return None

        def __getattr__(self, name):
            return getattr(self._inner, name)

    def _TR_patched_init(self, *args, **kwargs):
        _TR_orig_init(self, *args, **kwargs)
        try:
            self.replay_buffer = GuardedBuffer(self.replay_buffer, ACTION_SPACE_SIZE)
        except Exception:
            pass

    _Trainer.__init__ = _TR_patched_init

# ---------- Run original CLI if present ----------
if __name__ == "__main__":
    if hasattr(base, "main") and callable(getattr(base, "main")):
        sys.exit(base.main())
    # Fallback: if no main, just print info
    print("Patched mirrormind_agi imported. No `main()` found to run.")
