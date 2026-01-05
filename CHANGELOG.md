# Changelog

All notable changes to AirborneHRS are documented in this file.

## [1.1.1] - 2026-01-05 "Sentient" Edition

### 🚀 Major Features

- **System 2 Thinking**: Recursive Global Workspace for multi-step reasoning
- **Consciousness Core V2**: Emotional dynamics, metacognition, self-model, and personality
- **Unified Memory Handler**: Hybrid EWC + SI with OGD gradient projection
- **Gradient Centralization**: Improved optimization stability
- **Lookahead Optimizer**: Slow/fast weight synchronization

### ✅ SOTA Benchmarks (All Passed)

| Test | Description |
| :--- | :--- |
| Few-Shot | >30% improvement in 10 shots |
| Forgetting | Task A retained after Task B |
| Noise | Stable under Gaussian noise |
| OOD Detection | Surprise=128.9 for OOD inputs |
| System 2 | Adaptive thought trace depth |

### 🐛 Bug Fixes

- Fixed `mse_cpu` type mismatch during loss calculation
- Fixed `AssertionError: embed_dim must be divisible by num_heads`
- Fixed `UnboundLocalError: cannot access local variable 'logits'`
- Added missing `confusion` metric to consciousness output

### 📁 Files Modified

- `airbornehrs/core.py` - Complete V8.0 rewrite with train_step fixes
- `airbornehrs/consciousness_v2.py` - Added RecursiveGlobalWorkspace, confusion metric
- `airbornehrs/memory.py` - Fixed type casting in Fisher consolidation
- `verification_script.py` - Fixed model_dim/num_heads configuration
- `benchmark_sota.py` - New SOTA benchmarking suite

---

## [7.0.0] - Previous Release

- Unified Memory Handler (EWC + SI)
- MoE Transformation (Sparse Mixture of Experts)
- Basic Consciousness Layer

---

## [6.1.0] - Legacy

- Initial public release
- EWC-based memory protection
- Reptile meta-learning
