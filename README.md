# ♟️ MirrorMind Adv — A Groundbreaking Chess Transformer AI (One-File README)

> *“If AlphaZero was a revolution, this is a scalpel.”*

MirrorMind Adv is a publication-grade, research-oriented chess AI system built from the ground up using PyTorch. It rethinks how Transformers can be used in **spatially structured games** like chess by introducing **multi-scale attention**, **game-phase awareness**, and **self-supervised auxiliary learning** — all in one modular pipeline. This single README wraps the overview, deep-dive explanation, setup, usage, and roadmap into one file you can drop into any repo.

---

## ✨ Highlights

- 🧠 **Chess-Specific Positional Encoding** — rank, file, diagonal, and square color embeddings give the model geometric priors.
- 🔍 **Multi-Scale Attention** — local (windowed) + global attention streams capture tactics *and* strategy.
- 🕰️ **Game Phase Awareness** — explicit opening/middlegame/endgame encoding allows the policy to adapt over time.
- 🎯 **Dual-Head Architecture** — policy logits over 4,096 actions + scalar value head.
- 🧪 **Auxiliary Self-Supervised Tasks** — piece reconstruction & phase classification accelerate representation learning.
- ♻️ **Prioritized Experience Replay (PER)** — TD-error–based sampling with importance weights for efficient learning.
- 🔁 **TD-λ Returns** — blends bootstrapped and Monte Carlo targets to reduce variance.
- ⚡ **End-to-End CLI** — train, play vs human, analyze FENs, and benchmark against baselines.

---

## 🧭 Table of Contents

1. [Why It’s Groundbreaking](#-why-its-groundbreaking)
2. [Architecture at a Glance](#-architecture-at-a-glance)
3. [Detailed Modules](#-detailed-modules)
4. [Training Loop & RL Details](#-training-loop--rl-details)
5. [Modes & How to Use](#-modes--how-to-use)
6. [Hyperparameters](#-hyperparameters)
7. [Visualization](#-visualization)
8. [Performance Benchmarks](#-performance-benchmarks)
9. [Design Choices & Trade-offs](#-design-choices--trade-offs)
10. [Future Work](#-future-work)
11. [Setup & Dependencies](#-setup--dependencies)
12. [Cite This Work](#-cite-this-work)
13. [License](#-license)

---

## 💡 Why It’s Groundbreaking

Most chess AIs fall into two camps:

- **Search-first** (e.g., Stockfish): blazingly fast but hand-crafted evaluation functions.
- **AlphaZero-style**: powerful but compute-hungry (MCTS + deep nets + huge self-play).

**MirrorMind Adv** proposes a *third path*:
- A **Transformer** with **domain-aware inductive biases** (geometry + phases) that learns robust policies and values **without MCTS**.
- **Multi-scale attention** fuses **local** tactics (pins, forks, captures) with **global** strategy (king safety, pawn structure) in the same layer.
- **Auxiliary supervision** improves sample efficiency and stability, making **low-power learning** viable.

This enables practical, research-grade play and analysis **without** massive clusters — a big deal for reproducible research and on-device agents.

---

## 🧱 Architecture at a Glance

**Token space**
- 16 tokens total: 13 piece tokens (white 1–6, black 7–12), plus `EMPTY=0`, `CLS=13`, `MASK=14`, `ENDGAME=15`.
- Board flattened to 64 tokens; prepend a `CLS` token for sequence pooling.

**Embeddings & encoders**
- **Token embedding** + **PositionalEncoding** (learned) augmented with **rank/file/diagonal/color** embeddings for board geometry.
- **GamePhaseEncoder**: opening/middlegame/endgame + move count + normalized material to bias policy/value by phase.

**Core model**
- `num_layers` stacks of **ChessTransformerBlock** with **MultiScaleAttention** (local windowed + global full) and **gated feed-forward**.

**Heads**
- **Policy head** → logits over 4,096 (64×64) from-to moves (promotion simplified to queen auto-promotion for pawns).
- **Value head** → scalar evaluation.
- **Auxiliary heads** → piece reconstruction (board tokens) and phase classification (opening/middle/end).

---

## 🔬 Detailed Modules

### 1) PositionalEncoding (Chess-Aware)
- Standard positional embeddings for sequence index.
- Chess geometry embeddings:
  - **Rank** (0–7), **File** (0–7), **Diagonal** (0–14), **Color** (light/dark).
- Concatenated and projected to align with `d_model`, added to token embeddings.

**Why it matters:** Transformers are permutation-invariant by default; this injects **spatial structure** so bishops “feel” diagonals, rooks “feel” ranks/files, and central squares get modeled distinctly.

---

### 2) MultiScaleAttention
Two attention streams:
- **Local**: attention mask allows tokens within a Chebyshev window (default radius 5). Great for short-range tactics.
- **Global**: vanilla attention across all squares. Great for long-range strategy and global constraints.
- Outputs concatenated → linear projection to `d_model`.

**Why it matters:** Chess has **multi-scale dependencies**; encoding both explicitly improves data efficiency and interpretability.

---

### 3) ChessTransformerBlock
- Residual + LayerNorm wrappers.
- **Gated FFN**: `FF1(x) * σ(Gate(x)) → GELU → FF2`, improves selective feature flow.
- Dropout for regularization.

---

### 4) GamePhaseEncoder
- Detects phase using heuristics (move count & material) and embeds it.
- Adds phase vector into the `CLS` token representation.

**Why it matters:** The optimal policy changes by phase; giving the model phase context reduces mode collapse (e.g., overvaluing development in endgames).

---

### 5) Heads: Policy, Value, and Auxiliary
- **Policy**: MLP + LayerNorm → logits over **ACTION_SPACE_SIZE = 64×64**.
- **Value**: MLP + LayerNorm → scalar.
- **Auxiliary**:
  - **Piece reconstruction**: cross-entropy over 13 piece types per square (ignoring empty token).
  - **Phase prediction**: 3-class classification from `CLS` representation.

**Why it matters:** Auxiliary tasks provide **self-supervised signals** even when rewards are sparse or noisy, stabilizing early training.

---

## 🏋️ Training Loop & RL Details

- **Environment**: uses `python-chess` for legal move generation, board updates, outcomes; optional Stockfish eval for diagnostics.
- **Rewards**:
  - Checkmate ±10
  - Check ±0.5
  - Legal move +0.01
  - Illegal move −1 + random fallback legal move
  - Draw/stalemate/insufficient material → 0
- **Action selection**:
  - Temperature-controlled softmax over **masked** legal move logits.
  - Occasional random exploration via `exploration_noise`.
- **Replay**: **Prioritized Experience Replay** with α/β and importance sampling weights.
- **Targets**: **TD-λ returns** blend bootstrapped values with multi-step returns.
- **Loss**:
  - Policy loss = `−logπ(a|s) * (R − V)`
  - Value loss = MSE(`V`, `R`)
  - Aux loss (piece reconstruction) = CE
  - Entropy bonus to encourage exploration
- **Optimization**:
  - `AdamW` + cosine annealing LR
  - Gradient clipping (1.0)
  - Target network syncing by steps

---

## 🕹️ Modes & How to Use

Run the script and pick a mode from the interactive prompt:

```bash
python mirrormind_adv.py  # then choose one: train / play / analyze / benchmark
```

### 1) 🔁 Train
- Choose number of self-play games (e.g., 2000).
- Checkpoints saved periodically (e.g., every 200 games).
- Optional: load from existing checkpoint to resume.

### 2) 👤 Play vs Human
- Load a trained checkpoint, play as Black vs human White in **UCI** notation.
- Commands: `hint` (AI’s best move), `quit` to exit.

### 3) 🧠 Analyze
- Load a trained checkpoint, paste any **FEN** string.
- Returns:
  - Top-5 moves (by policy logits)
  - Scalar evaluation
  - Phase probabilities (Opening/Middlegame/Endgame)

### 4) 🧪 Benchmark
- Pits the model vs two baselines:
  - **Random** move generator
  - **Greedy** (prefer captures/checks)
- Reports W/L/D counts and win-rates.

---

## ⚙️ Hyperparameters

| Parameter             | Default | Notes |
|----------------------|---------|-------|
| `d_model`            | 256     | Embedding width |
| `nhead`              | 16      | Attention heads |
| `num_layers`         | 8       | Transformer depth |
| `dropout`            | 0.1     | Regularization |
| `learning_rate`      | 3e-4    | AdamW LR |
| `batch_size`         | 32      | Batch size for training |
| `memory_size`        | 100000  | PER capacity |
| `target_update_freq` | 1000    | Target net sync steps |
| `exploration_noise`  | 0.3     | ε for random exploration |
| `temperature`        | 1.0     | Softmax temperature |
| `lambda_value`       | 0.95    | TD-λ weight |
| `entropy_coeff`      | 0.01    | Entropy regularization |

> **Publication-grade config (example in `main()`):** `d_model=384, nhead=24, num_layers=12, lr=1e-4, batch_size=64, memory_size=200000, exploration_noise=0.2, temperature=0.8`

---

## 📈 Visualization

Use the built-in utility to plot **rewards**, **policy loss**, **value loss**, and **aux loss**. It will save `training_progress.png`. Requires `matplotlib`.

---

## 🧪 Performance Benchmarks

The included benchmarking harness reports:
- **Wins/Losses/Draws**
- **Win-rate (%)**

Opponents:
- **Random**
- **Greedy** (capture/check heuristic)

> This is meant as a **sanity check** and regression test. For rigorous evaluation, add ELO pools, opening books, or match vs classical engines under fixed time controls.

---

## 🧠 Design Choices & Trade-offs

- **No MCTS**: We prioritize *architectural priors* over search to enable **low-compute** training and inference.
- **Action space 64×64**: Simple and universal, but promotions are simplified (auto-queen); extend with underpromotions if needed.
- **Heuristic phase detection**: Lightweight and surprisingly effective; can be replaced with a learned phase estimator.
- **Aux tasks**: Great for stability; keep weights tuned to avoid overpowering the main RL signal.

---

## 🔮 Future Work

- [ ] Curriculum learning (openings → endgames).
- [ ] Integrate **MCTS** optionally for strength at higher compute.
- [ ] Pretrain on large human PGN datasets for better priors.
- [ ] Export to ONNX + quantization for mobile.
- [ ] Richer action head for promotions/castling disambiguation.
- [ ] Add ELO evaluation pipelines.

---

## 🔧 Setup & Dependencies

```bash
pip install torch numpy python-chess matplotlib
# Optional: install stockfish and ensure it’s on PATH if you want engine evals.
```

**Run:**

```bash
python mirrormind_adv.py
# Then choose: train / play / analyze / benchmark
```

---

## 📚 Cite This Work

If you use MirrorMind Adv in research or production, please cite:

```
@software{mirrormind_adv_2025,
  title  = {MirrorMind Adv: Multi-Scale Transformer Chess Agent with Game-Phase Awareness},
  author = {Anonymous},
  year   = {2025},
  note   = {https://github.com/your-repo/mirrormind-adv}
}
```

---

## ⚖️ License

MIT License — free to use, modify, and distribute with attribution.
