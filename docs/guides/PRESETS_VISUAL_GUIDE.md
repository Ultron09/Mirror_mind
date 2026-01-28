# ANTARA Presets - Visual Comparison Guide

## Preset Comparison Matrix

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                          MIRRORMIMD PRESETS OVERVIEW                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│ PRODUCTION                                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ Best for: Real applications, high accuracy, production deployment           │
│ Model Size: 512 (Large, expressive)                                         │
│ Learning Rate: 5e-4 (Conservative, stable)                                  │
│ Buffer Size: 20,000 (Extensive memory)                                      │
│ Memory Type: Hybrid (EWC + SI)                                              │
│ Consciousness: ENABLED (5D awareness)                                       │
│ Speed: ████░░░░░░ Medium (100 ops/sec)                                      │
│ Accuracy: ██████████ Excellent (95%+)                                       │
│ Memory: ███████░░░ High (2GB)                                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ BALANCED (Recommended Starting Point)                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ Best for: General purpose, development, safe default                        │
│ Model Size: 256 (Medium)                                                    │
│ Learning Rate: 1e-3 (Moderate)                                              │
│ Buffer Size: 10,000 (Standard)                                              │
│ Memory Type: Hybrid                                                         │
│ Consciousness: ENABLED                                                      │
│ Speed: ██████░░░░ Good (500 ops/sec)                                        │
│ Accuracy: █████░░░░░ Good (90%+)                                            │
│ Memory: ██████░░░░ Moderate (1GB)                                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ FAST                                                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ Best for: Real-time, robotics, streaming                                    │
│ Model Size: 128 (Small)                                                     │
│ Learning Rate: 5e-3 (Aggressive)                                            │
│ Buffer Size: 2,000 (Minimal)                                                │
│ Memory Type: SI (Lightweight)                                               │
│ Consciousness: DISABLED (saves cycles)                                      │
│ Speed: ██████████ Excellent (5K ops/sec)                                    │
│ Accuracy: ███░░░░░░░ Fair (80%+)                                            │
│ Memory: ██░░░░░░░░ Minimal (200MB)                                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ ACCURACY_FOCUS                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ Best for: Medical, finance, high-consequence                                │
│ Model Size: 512 (Large)                                                     │
│ Learning Rate: 1e-4 (Very conservative)                                     │
│ Buffer Size: 50,000 (Extensive)                                             │
│ Memory Type: EWC (Proven stable)                                            │
│ Consciousness: ENABLED                                                      │
│ Speed: ██░░░░░░░░ Slow (50 ops/sec)                                         │
│ Accuracy: ██████████ Maximum (98%+)                                         │
│ Memory: ███████████ Very High (3GB)                                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ MEMORY_EFFICIENT                                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ Best for: Mobile, edge devices, IoT                                         │
│ Model Size: 64 (Tiny)                                                       │
│ Learning Rate: 1e-3                                                         │
│ Buffer Size: 1,000 (Minimal)                                                │
│ Memory Type: SI                                                             │
│ Consciousness: Lightweight                                                  │
│ Speed: ███████░░░░ Fast (2K ops/sec)                                        │
│ Accuracy: ██░░░░░░░░ Lower (75%+)                                           │
│ Memory: █░░░░░░░░░ Minimal (100MB)                                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ EXPLORATION                                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ Best for: Curiosity-driven learning, multi-task                             │
│ Model Size: 384 (Large)                                                     │
│ Learning Rate: 2e-3 (Exploratory)                                           │
│ Buffer Size: 15,000                                                         │
│ Memory Type: Hybrid                                                         │
│ Consciousness: ENABLED (intrinsic motivation HIGH)                          │
│ Speed: █████░░░░░ Good (200 ops/sec)                                        │
│ Accuracy: █████░░░░░ Good (88%+)                                            │
│ Memory: ██████░░░░ Moderate (1.5GB)                                         │
│ Key Feature: Novelty detection enabled                                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ CREATIVITY_BOOST                                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ Best for: Generative, creative applications                                 │
│ Model Size: 256                                                             │
│ Learning Rate: 1.5e-3                                                       │
│ Buffer Size: 12,000                                                         │
│ Memory Type: Hybrid                                                         │
│ Dropout: 0.25 (High for diversity)                                          │
│ Speed: ██████░░░░ Good (400 ops/sec)                                        │
│ Accuracy: █████░░░░░ Good (87%+)                                            │
│ Diversity: ██████████ Maximum                                               │
│ Key Feature: Soft priority sampling                                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STABLE                                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ Best for: Safety-critical, maximum reliability                              │
│ Model Size: 512 (Large)                                                     │
│ Learning Rate: 5e-4 (Conservative)                                          │
│ Buffer Size: 30,000                                                         │
│ Memory Type: EWC ONLY (proven stable)                                       │
│ Consciousness: Minimal (no attention)                                       │
│ Speed: ████░░░░░░ Medium (80 ops/sec)                                       │
│ Robustness: ██████████ Maximum                                              │
│ Forgetting: █░░░░░░░░░ Minimal                                              │
│ Key Feature: Catastrophic forgetting prevention                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ RESEARCH                                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ Best for: Papers, ablation studies, experimentation                         │
│ Model Size: 256 (Standard)                                                  │
│ Learning Rate: 1e-3                                                         │
│ Tracing: ENABLED (all metrics logged)                                       │
│ Checkpointing: FREQUENT (every 100 steps)                                   │
│ All Features: ENABLED (full instrumentation)                                │
│ Speed: ██████░░░░ Good (500 ops/sec)                                        │
│ Observability: ██████████ Maximum                                           │
│ Key Feature: Full metric tracking                                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ REAL_TIME                                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│ Best for: Sub-millisecond inference, streaming                              │
│ Model Size: 96 (Very small)                                                 │
│ Learning Rate: 2e-3                                                         │
│ Buffer Size: 1,500                                                          │
│ Memory Type: SI (Fast)                                                      │
│ Consciousness: Very lightweight                                             │
│ Speed: ███████████ Extreme (10K ops/sec)                                    │
│ Latency: █░░░░░░░░░ Sub-millisecond                                         │
│ Key Feature: Minimal overhead                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Decision Tree

```
                          Choose a Preset
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
              Production?    Speed Critical?  Creative?
                /  │  \         /    \        /  │  \
              YES  NO  ?       YES    NO     YES NO  MAYBE
              │    │   │        │     │      │   │   │
              ↓    │   │        ↓     │      ↓   │   │
          PROD    │   │       FAST    │    CREA  │  EXPL
          OR      │   │        OR   REAL_TIME  │
          ACCY    │   │       R_TIME            │
                  │   │                        │
                  └─→ BALANCED ←────────────────┘
                         │
                  ┌──────┼──────┐
                  │      │      │
            Memory?   High?   Safety?
            Limit?   Accuracy? Critical?
             │  │     │  │      │  │
             Y  N     Y  N      Y  N
             │  │     │  │      │  │
             ↓  │     ↓  │      ↓  │
           MEM  │    ACCY │    STABLE │
                │         │          │
                └─────────┴──────────┴→ RESEARCH
```

---

## Feature Comparison Table

```
╔═══════════════════════╦════════╦═══════╦═════════╦═════════╦═════════════╗
║ Preset                ║ Speed  ║ Accuracy║ Memory ║ Conscious║ Memory Type║
╠═══════════════════════╬════════╬═══════╬═════════╬═════════╬═════════════╣
║ PRODUCTION            ║ ████   ║ ██████ ║ ███████ ║ YES     ║ Hybrid      ║
║ BALANCED              ║ █████  ║ █████  ║ ██████  ║ YES     ║ Hybrid      ║
║ FAST                  ║ █████│ ║ ███    ║ ██      ║ NO      ║ SI          ║
║ ACCURACY_FOCUS        ║ ██     ║ ██████ ║ ██████  ║ YES     ║ EWC         ║
║ MEMORY_EFFICIENT      ║ ██████ ║ ██     ║ █       ║ LITE    ║ SI          ║
║ EXPLORATION           ║ █████  ║ █████  ║ ██████  ║ YES+    ║ Hybrid      ║
║ CREATIVITY_BOOST      ║ █████  ║ █████  ║ ██████  ║ YES     ║ Hybrid      ║
║ STABLE                ║ ████   ║ █████  ║ ███████ ║ LITE    ║ EWC         ║
║ RESEARCH              ║ █████  ║ █████  ║ ██████  ║ YES     ║ Hybrid      ║
║ REAL_TIME             ║ ██████ ║ ████   ║ █       ║ LITE    ║ SI          ║
╚═══════════════════════╩════════╩═══════╩═════════╩═════════╩═════════════╝
```

---

## Hyperparameter Ranges

```
Learning Rate Distribution:
├─ 1e-4      ████░░░░░░ ACCURACY_FOCUS (conservative)
├─ 5e-4      ██████░░░░ PRODUCTION, STABLE (careful)
├─ 1e-3      ████████░░ BALANCED, RESEARCH (moderate)
├─ 1.5e-3    ███████░░░ CREATIVITY_BOOST
├─ 2e-3      ███████░░░ EXPLORATION, REAL_TIME (fast)
├─ 2e-3      ███████░░░
└─ 5e-3      ██████░░░░ FAST (aggressive)

Model Dimensions Distribution:
├─ 64        █░░░░░░░░░ MEMORY_EFFICIENT (tiny)
├─ 96        ██░░░░░░░░ REAL_TIME (small)
├─ 128       ███░░░░░░░ FAST
├─ 256       █████░░░░░ BALANCED, CREATIVITY, RESEARCH (medium)
├─ 384       ██████░░░░ EXPLORATION (large)
└─ 512       ███████░░░ PRODUCTION, ACCURACY, STABLE (xl)

Buffer Size Distribution:
├─ 1,000     █░░░░░░░░░ MEMORY_EFFICIENT (minimal)
├─ 1,500     █░░░░░░░░░ REAL_TIME
├─ 2,000     ██░░░░░░░░ FAST
├─ 10,000    █████░░░░░ BALANCED, RESEARCH (standard)
├─ 12,000    ██████░░░░ CREATIVITY_BOOST
├─ 15,000    ███████░░░ EXPLORATION
├─ 20,000    ████████░░ PRODUCTION (large)
└─ 50,000    ██████████ ACCURACY_FOCUS (xl)
```

---

## Use Case Mapping

```
Medical/Healthcare
├─ Accuracy CRITICAL → ACCURACY_FOCUS ✓
├─ Safety CRITICAL → STABLE
└─ Speed Optional → PRODUCTION is OK

Robotics/Real-time
├─ Latency < 10ms → FAST or REAL_TIME ✓
├─ Some accuracy OK → FAST
└─ Need learning → FAST is better

Text/Image Generation
├─ Diversity important → CREATIVITY_BOOST ✓
├─ Speed matters → CREATIVITY_BOOST
└─ Accuracy secondary → CREATIVITY_BOOST

Mobile/Edge
├─ Memory < 500MB → MEMORY_EFFICIENT ✓
├─ Latency critical → REAL_TIME
└─ Power limited → MEMORY_EFFICIENT

Research
├─ Need all metrics → RESEARCH ✓
├─ Publishing paper → RESEARCH
└─ Ablation study → RESEARCH

Financial/Risk
├─ Accuracy CRITICAL → ACCURACY_FOCUS ✓
├─ Stability CRITICAL → STABLE
└─ Performance OK → ACCURACY_FOCUS

General Purpose
├─ Don't know use case → BALANCED ✓
├─ Learn and adjust → BALANCED
└─ Good starting point → BALANCED

Safety-Critical
├─ Zero failure margin → STABLE ✓
├─ Proven algorithms → STABLE (EWC only)
└─ Conservative → STABLE
```

---

## Performance Profile

```
Speed vs Accuracy Trade-off:

ACCURACY
   ↑
   │  ██ ACCURACY_FOCUS (98%)
   │
   │  ████ PRODUCTION (95%)
   │  ████ BALANCED (90%)
   │  ████ EXPLORATION (88%)
   │  ████ CREATIVITY (87%)
   │  ████ RESEARCH (90%)
   │  ████ STABLE (93%)
   │
   │  ██ FAST (80%)
   │
   │  ██ MEMORY_EFFICIENT (75%)
   │  ██ REAL_TIME (82%)
   │
   └─────────────────────────────────→ SPEED
        SLOW    MEDIUM    FAST    VERY FAST

   Best Overall: PRODUCTION (sweet spot)
   Best Accuracy: ACCURACY_FOCUS
   Best Speed: REAL_TIME
   Best Balance: BALANCED
```

---

## Quick Recommendation

```
"What preset should I use?"

┌─ Do you know what you're doing?
│
├─ NO → Start with BALANCED
│   └─ Too slow? → Switch to FAST
│   └─ Not accurate? → Switch to ACCURACY_FOCUS
│   └─ Too much memory? → Switch to MEMORY_EFFICIENT
│
└─ YES → Depends on your goal:
   ├─ Maximum accuracy → ACCURACY_FOCUS
   ├─ Production deployment → PRODUCTION
   ├─ Real-time/robotics → FAST or REAL_TIME
   ├─ Mobile/Edge → MEMORY_EFFICIENT
   ├─ Creative/diverse → CREATIVITY_BOOST
   ├─ Curiosity-driven → EXPLORATION
   ├─ Safety-critical → STABLE
   ├─ Research/paper → RESEARCH
   └─ Still not sure → PRODUCTION (safest bet)
```

---

## One-Liner Cheatsheet

```python
# Copy-paste these

# Production-ready
from airbornehrs import AdaptiveFramework, PRESETS
fw = AdaptiveFramework(model, config=PRESETS.production())

# Real-time
fw = AdaptiveFramework(model, config=PRESETS.fast())

# Maximum accuracy
fw = AdaptiveFramework(model, config=PRESETS.accuracy_focus())

# Mobile/edge
fw = AdaptiveFramework(model, config=PRESETS.memory_efficient())

# Creative
fw = AdaptiveFramework(model, config=PRESETS.creativity_boost())

# Curious exploration
fw = AdaptiveFramework(model, config=PRESETS.exploration())

# Safety-critical
fw = AdaptiveFramework(model, config=PRESETS.stable())

# Research/benchmarking
fw = AdaptiveFramework(model, config=PRESETS.research())

# Don't know yet (best default)
fw = AdaptiveFramework(model, config=PRESETS.balanced())
```

---

That's it! Pick one based on your use case and you're ready to go. 🚀
