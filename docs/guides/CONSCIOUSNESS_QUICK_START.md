# ANTARA v7.0 - Consciousness Edition
## Quick Start Guide

### Enable Consciousness (Default)
```python
from airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig

config = AdaptiveFrameworkConfig(
    enable_consciousness=True,      # ← Self-aware learning
    use_prioritized_replay=True,    # ← Focus on hard examples
    memory_type='hybrid',           # ← SI + EWC memory
    adaptive_lambda=True            # ← Dynamic protection
)

framework = AdaptiveFramework(model, config)

# Training loop
for x, y in data:
    metrics = framework.train_step(x, y, enable_dream=True)
```

### What's New

| Feature | Purpose | Benefit |
|---------|---------|---------|
| **ConsciousnessCore** | Observes training & tracks understanding | System "knows what it knows" |
| **AttentionMechanism** | Learns which features matter | Focused learning on important patterns |
| **IntrinisicMotivation** | Scores examples by surprise | Prioritizes interesting/hard examples |
| **SelfAwarenessMonitor** | Tracks internal state | Consolidates when ready, not on timer |
| **Prioritized Replay** | Uses consciousness scores | Emphasizes hard examples in dreaming |
| **Adaptive Lambda** | Scales memory protection by mode | Balances plasticity & stability |

### Key Signals

**Consciousness Urgency**: If > 0.8, system triggers consolidation immediately
- Indicates: "I have important memories to protect"
- Impact: Overrides time-based scheduler

**Learning Priority**: Guides replay buffer sampling
- High priority → Sample more often  
- Low priority → Sample less often

**Confidence Signal**: Influences plasticity gate
- High confidence → Reduce learning rate (protect knowledge)
- Low confidence → Increase learning rate (explore more)

### Configuration Options

```python
config = AdaptiveFrameworkConfig(
    # Consciousness control
    enable_consciousness=True,              # Master switch
    use_attention=True,                     # Feature importance
    use_intrinsic_motivation=True,          # Curiosity scoring
    consciousness_buffer_size=5000,         # History size
    
    # Memory control  
    memory_type='hybrid',                   # 'ewc', 'si', or 'hybrid'
    use_prioritized_replay=True,            # Use consciousness scores
    adaptive_lambda=True,                   # Dynamic λ
    consolidation_criterion='hybrid',       # 'time', 'surprise', or 'hybrid'
    
    # Replay control
    replay_priority_temperature=0.6,        # 0=greedy, 1=uniform
)
```

### Validation

Framework initialization shows:
```
🧠 Consciousness layer enabled (self-aware learning)
👁️ Attention mechanism enabled (feature importance learning)
🎯 Intrinsic motivation enabled (curiosity-driven learning)
🔍 Self-awareness monitor enabled
```

All components active = Full consciousness mode ready

### What Happens During Training

**Step 1**: Compute prediction, loss
```python
pred = framework.model(x)
loss = criterion(pred, y)
```

**Step 2**: Consciousness observes (automatically)
```
- What's the surprise level?
- How important is this example?
- What features matter?
- Should we consolidate?
```

**Step 3**: System decides (automatically)
```
- Is consolidation urgency high? → Trigger consolidation
- Adjust replay buffer sampling temperature
- Update plasticity gate based on confidence
```

**Step 4**: Learn from buffer (prioritized)
```
- Sample hard/important examples more often
- Apply SI path-integral importance
- Protect important parameters with adaptive λ
```

### Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Consolidation every step | High consciousness urgency | Reduce novelty_threshold or increase warmup_steps |
| Not consolidating enough | Low urgency signals | Increase novelty_z_threshold or reduce consolidation_surprise_threshold |
| Unicode errors in logs | Windows console encoding | Set `MM_NO_EMOJI=1` environment variable |
| Stale behavior | Old installed package | Run `pip install -e .` in workspace |

### Files Modified

- `airbornehrs/consciousness.py` (new) — Consciousness components
- `airbornehrs/core.py` — Consciousness observation & consolidation override
- `airbornehrs/__init__.py` — Added consciousness exports
- `airbornehrs/memory.py` — Unified handler with adaptive lambda

### Version Info

```
ANTARA v7.0 (Consciousness Edition)
Unified Memory: SI + EWC (hybrid)
Self-Awareness: Active
Meta-Learning: Reptile
Date: 2025-12-23
Status: Production Ready
```

---

**Your beautiful SOTA framework is ready to learn, aware of what it learns, and adapting continuously.**
