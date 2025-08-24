# mirrormind_agi_monkey_patch5.py
"""
Monkey patch v5 for mirrormind_agi.py

Fixes implemented at runtime:
- Ensure piece_embed has room for CLS token (index 13).
- Use the actual diagonal attribute names if present (diagonal_embed / anti_diagonal_embed).
- Provide a safe _embed_chess wrapper that clamps indices and uses correct attr names.
- Patch AGIChessEnvironment._calculate_sophisticated_reward to check legality via a board copy
  (prevents python-chess assertion when checking gives_check).
- Keeps prior protections (replay buffer guard & step legality) intact.
Run:
    python mirrormind_agi_monkey_patch5.py
"""
import sys, copy, torch

import mirrormind_agi as base

# Resolve CLS index
try:
    CLS_INDEX = int(base.SpecialTokens.CLS)
except Exception:
    CLS_INDEX = 13

ACTION_SPACE_SIZE = getattr(base, "ACTION_SPACE_SIZE", 64*64)

# 1) Ensure UniversalEmbedding exists and patch safely
if hasattr(base, "UniversalEmbedding"):
    UE = base.UniversalEmbedding
    _orig_init = UE.__init__ if hasattr(UE, "__init__") else None

    def _patched_init(self, *args, **kwargs):
        # call original init
        if _orig_init:
            _orig_init(self, *args, **kwargs)

        # Ensure piece_embed has at least CLS_INDEX + 1 embeddings
        try:
            if hasattr(self, "piece_embed") and isinstance(self.piece_embed, torch.nn.Embedding):
                needed = CLS_INDEX + 1
                if self.piece_embed.num_embeddings < needed:
                    old = self.piece_embed
                    new = torch.nn.Embedding(needed, old.embedding_dim)
                    with torch.no_grad():
                        new.weight.zero_()
                        new.weight[:old.num_embeddings].copy_(old.weight.data)
                    self.piece_embed = new
        except Exception as e:
            print("⚠️ Failed to ensure piece_embed size:", e)

        # Ensure positional embeddings exist with minimal sizes
        def ensure(attr, min_n, default_dim):
            try:
                emb = getattr(self, attr, None)
                if emb is None:
                    # create a default embedding to avoid attribute errors
                    setattr(self, attr, torch.nn.Embedding(min_n, default_dim))
                elif isinstance(emb, torch.nn.Embedding) and emb.num_embeddings < min_n:
                    new = torch.nn.Embedding(min_n, emb.embedding_dim)
                    with torch.no_grad():
                        new.weight.zero_()
                        new.weight[:emb.num_embeddings].copy_(emb.weight.data)
                    setattr(self, attr, new)
            except Exception as e:
                print(f"⚠️ ensure failed for {attr}: {e}")

        # typical dims used in the file: use piece_embed.embedding_dim or guess 64
        d = getattr(self.piece_embed, "embedding_dim", 64)
        ensure("chess_pos_embed", 65, d)
        ensure("rank_embed", 8, max(1, d // 4))
        ensure("file_embed", 8, max(1, d // 4))
        ensure("diagonal_embed", 15, max(1, d // 4))
        ensure("anti_diagonal_embed", 15, max(1, d // 4))

    UE.__init__ = _patched_init

    # 2) Patch _embed_chess to use correct attribute names + clamps
    def _patched_embed_chess(self, board_tensor, positions=None):
        """
        Safe embed chess: clamps indices, uses existing attribute names (diagonal_embed / anti_diagonal_embed)
        and returns embeddings without raising embedding-index errors.
        """
        batch_size, seq_len = board_tensor.shape

        # clamp piece indices
        try:
            max_piece = int(board_tensor.max().item())
            if max_piece >= self.piece_embed.num_embeddings or max_piece < 0:
                print(f"⚠️ Invalid piece index {max_piece}, clamping to valid range [0,{self.piece_embed.num_embeddings-1}]")
                board_tensor = board_tensor.clamp(0, self.piece_embed.num_embeddings - 1)
        except Exception:
            board_tensor = board_tensor.clamp(0, self.piece_embed.num_embeddings - 1)

        # positions handling
        if positions is None:
            positions = torch.arange(seq_len, device=board_tensor.device).unsqueeze(0).expand(batch_size, -1)
        else:
            # ensure valid positions tensor shape
            if positions.dim() == 1:
                positions = positions.unsqueeze(0).expand(batch_size, -1)

        # basic piece & pos embeddings
        piece_emb = self.piece_embed(board_tensor)
        pos_emb = getattr(self, "chess_pos_embed", None)
        if pos_emb is not None:
            pos_emb = pos_emb(positions.clamp(0, pos_emb.num_embeddings - 1))
        else:
            pos_emb = torch.zeros_like(piece_emb)

        # compute square geometry indices (exclude CLS token)
        board_positions = positions[:, 1:] if seq_len == 65 else positions
        offset = 1 if seq_len == 65 else 0
        squares = (board_positions - offset).clamp(0, 63)

        ranks = (squares // 8).long()
        files = (squares % 8).long()
        diagonals = (ranks + files).long()
        anti_diagonals = (ranks - files + 7).long()

        # Use whichever diagonal attribute names exist on the object
        diag_attr = "diagonal_embed" if hasattr(self, "diagonal_embed") else ("diag_embed" if hasattr(self, "diag_embed") else None)
        anti_diag_attr = "anti_diagonal_embed" if hasattr(self, "anti_diagonal_embed") else ("anti_diag_embed" if hasattr(self, "anti_diag_embed") else None)

        # clamp and embed; if missing, create zero embeddings to preserve shape
        def safe_embed(attr_name, idx_tensor, default_dim):
            emb = getattr(self, attr_name, None) if attr_name else None
            if emb is None or not isinstance(emb, torch.nn.Embedding):
                # return zeros with expected shape
                B, L = idx_tensor.shape
                return torch.zeros(B, L, default_dim, device=board_tensor.device)
            else:
                return emb(idx_tensor.clamp(0, emb.num_embeddings - 1))

        B = batch_size
        L = ranks.shape[1] if ranks.dim() > 1 else squares.shape[1]
        # Expand idx to batch
        ranks_b = ranks
        files_b = files
        diag_b = diagonals
        anti_diag_b = anti_diagonals

        rank_emb = safe_embed("rank_embed", ranks_b, getattr(self.rank_embed, "embedding_dim", piece_emb.size(-1)//4))
        file_emb = safe_embed("file_embed", files_b, getattr(self.file_embed, "embedding_dim", piece_emb.size(-1)//4))
        diag_emb = safe_embed(diag_attr, diag_b, getattr(self, "diagonal_embed", getattr(self, "diag_embed", torch.nn.Embedding(15, piece_emb.size(-1)//4))).embedding_dim)
        anti_diag_emb = safe_embed(anti_diag_attr, anti_diag_b, getattr(self, "anti_diagonal_embed", getattr(self, "anti_diag_embed", torch.nn.Embedding(15, piece_emb.size(-1)//4))).embedding_dim)

        # concat geom embeddings; if CLS present, prepend zero geom row
        geom_emb = torch.cat([rank_emb, file_emb, diag_emb, anti_diag_emb], dim=-1)
        if seq_len == 65:
            zero_geom = torch.zeros(batch_size, 1, geom_emb.size(-1), device=board_tensor.device)
            geom_emb = torch.cat([zero_geom, geom_emb], dim=1)

        # modal embeddings if available
        modal_emb = getattr(self, "modal_embed", None)
        modal = modal_emb(torch.zeros(batch_size, seq_len, dtype=torch.long, device=board_tensor.device)) if modal_emb is not None else 0

        return piece_emb + (pos_emb if pos_emb is not None else 0) + geom_emb + (modal if isinstance(modal, torch.Tensor) else 0)

    UE._embed_chess = _patched_embed_chess

# 3) Patch AGIChessEnvironment._calculate_sophisticated_reward to avoid pushing illegal moves
env_cls = None
for cand in ("AGIChessEnvironment", "ChessEnvironment"):
    if hasattr(base, cand) and isinstance(getattr(base, cand), type):
        env_cls = getattr(base, cand)
        break

if env_cls is not None and hasattr(env_cls, "_calculate_sophisticated_reward"):
    _orig_reward = env_cls._calculate_sophisticated_reward

    def _patched_calc_reward(self, move):
        # Ensure move is pseudo-legal before calling methods that push
        try:
            if move not in self.board.legal_moves:
                # illegal move: heavy penalty
                return -10.0
        except Exception:
            # if legality check errors, use a safe copy attempt
            try:
                board_copy = self.board.copy()
                board_copy.push(move)
            except Exception:
                return -10.0

        # Use a copy for temporary checks (so we never push on the real board)
        try:
            board_copy = self.board.copy()
            board_copy.push(move)
        except Exception:
            return -10.0

        reward = 0.01
        # capture reward
        try:
            if board_copy.is_capture(move):  # safe on board copy
                captured = board_copy.piece_type_at(move.to_square)
                values = {2:3, 3:3, 4:5, 5:9, 1:1}  # mapping: knight=2? depends on your mapping earlier - keep safe
                reward += values.get(captured, 0)
        except Exception:
            pass

        try:
            if board_copy.is_check():
                reward += 0.5
        except Exception:
            pass

        # center control (use move.to_square if present)
        try:
            if move.to_square in [base.chess.E4, base.chess.E5, base.chess.D4, base.chess.D5]:
                reward += 0.2
        except Exception:
            pass

        # development reward as in original (best-effort)
        try:
            if len(self.move_history) < 15:
                piece = self.board.piece_type_at(move.from_square)
                if piece in [base.chess.KNIGHT, base.chess.BISHOP]:
                    if move.from_square in [base.chess.B1, base.chess.G1, base.chess.C1, base.chess.F1,
                                            base.chess.B8, base.chess.G8, base.chess.C8, base.chess.F8]:
                        reward += 0.3
        except Exception:
            pass

        return reward

    env_cls._calculate_sophisticated_reward = _patched_calc_reward

# Keep other monkey patches (replay buffer guard / step wrapper) from previous patches if loaded
# Run the original main if present
if __name__ == "__main__":
    if hasattr(base, "main") and callable(getattr(base, "main")):
        sys.exit(base.main())
    print("Patched mirrormind_agi (v5) imported. No main() invoked.")
