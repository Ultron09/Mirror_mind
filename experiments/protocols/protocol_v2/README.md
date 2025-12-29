# Protocol v2: Comprehensive Testing & Validation Framework

## Overview

Protocol v2 is a comprehensive testing suite for MirrorMind v7.0, covering **8 distinct testing dimensions** with publication-ready visualizations and reports.

## Testing Dimensions

### 1. **Integration Testing** (`test_integration.py`)
Validates all core components work together:
- ✓ Consciousness observation and tracking
- ✓ Consolidation triggers and task memory saving
- ✓ Prioritized replay buffer functionality
- ✓ Memory protection (SI+EWC hybrid handler)
- ✓ Adaptive lambda scaling across modes
- ✓ End-to-end training with dreaming/meta-learning

### 2. **Usability Testing** (`test_usability.py`)
Ensures the framework is developer-friendly:
- ✓ Simple API with minimal code (~6 lines to train)
- ✓ Sensible defaults (consciousness ON, hybrid memory, prioritized replay)
- ✓ Robust error handling (shape/type validation)
- ✓ Flexible configuration (SI-only, no-consciousness, production modes)
- ✓ Complete documentation
- ✓ Informative logging

### 3. **Baseline Comparison** (`test_baselines.py`)
Compares against standard approaches:
- **Base Model**: Vanilla PyTorch (no memory/consciousness)
- **EWC-Only**: Elastic Weight Consolidation alone
- **SI-Only**: Synaptic Intelligence alone
- **MirrorMind v7.0**: Full hybrid memory + consciousness + prioritized replay + adaptive lambda

Metrics:
- Loss curves over 100 training steps
- Final loss comparison
- Learning improvement percentage
- Performance vs base model

### 4. **Multi-Modality Testing** (`test_multimodality.py`)
Validates support for diverse input types:
- **Vision**: 784D (MNIST-like images), 50 steps
- **Text**: 768D embeddings (BERT-like), 50 steps, binary output
- **Mixed**: 1040D (vision + text combined), 50 steps
- **High-dimensional**: 4096D inputs, 30 steps
- **Time-series**: 400D flattened sequences (50×8), 50 steps

### 5. **Memory Stress Testing** (`test_memory_stress.py`)
Verifies system stability under extreme memory conditions:
- **Large Replay Buffer**: 1,000 training steps with 10K sample capacity
- **Frequent Consolidation**: 100 steps with min_interval=1
- **Memory Retrieval**: 2-phase training, task memory persistence
- **Memory Efficiency**: psutil measurements, <500MB growth target
- **Prioritization Correctness**: Easy + hard sample weighting

### 6. **Adaptation Extremes Testing** (`test_adaptation_extremes.py`)
Tests learning in challenging scenarios:
- **Rapid Task Switching**: 5 tasks × 20 steps each, forgetting ratio measurement
- **Domain Shift**: Standard distribution → 10x scaled distribution
- **Continual Learning**: 10 sequential tasks × 30 steps each
- **Concept Drift**: Gradual scale increase (1.0 → 6.0) over 100 steps

### 7. **Survival Scenario Testing** (`test_survival_scenarios.py`)
Validates robustness and error recovery:
- **Panic Mode**: Normal (30 steps) → Crisis (20×10 scale) → Recovery (30 steps)
- **Sustained Load**: 200+ steps with random batch sizes (4-32)
- **Error Recovery**: Shape mismatches, NaN values, graceful handling
- **System Persistence**: Checkpoint save/load, buffer persistence

### 8. **Visualization & Reporting** (`visualization_reporter.py`)
Generates publication-ready outputs:
- **Loss curves** for baseline comparisons
- **Performance bars** with value labels
- **Multi-modality heatmaps**
- **Adaptation metrics plots**
- **Summary markdown reports**

## Running the Tests

### Option 1: Quick Start (Automatic)
```bash
cd experiments/protocol_v2
python quick_start.py
```

This runs all tests, generates visualizations, and produces a summary.

### Option 2: Master Runner (Detailed Control)
```bash
cd experiments/protocol_v2
python run_protocol_v2.py
```

Runs each test suite sequentially, saves individual JSON results, and creates an aggregated summary.

### Option 3: Individual Test Suites
```bash
cd experiments/protocol_v2/tests
python test_integration.py
python test_usability.py
python test_baselines.py
python test_multimodality.py
python test_memory_stress.py
python test_adaptation_extremes.py
python test_survival_scenarios.py
```

Each test produces a `{test_name}_results.json` file in `../results/`.

### Option 4: Generate Visualizations Only
```bash
cd experiments/protocol_v2
python visualization_reporter.py
```

Reads existing JSON results and generates plots/reports.

## Output Structure

```
experiments/protocol_v2/
├── tests/
│   ├── test_integration.py          # 6 integration tests
│   ├── test_usability.py            # 6 usability tests
│   ├── test_baselines.py            # 4 baseline comparisons
│   ├── test_multimodality.py        # 5 modality tests
│   ├── test_memory_stress.py        # 5 stress tests
│   ├── test_adaptation_extremes.py  # 4 extreme tests
│   └── test_survival_scenarios.py   # 4 survival tests
├── results/
│   ├── integration_test_results.json
│   ├── usability_test_results.json
│   ├── baseline_comparison_results.json
│   ├── multimodality_test_results.json
│   ├── memory_stress_test_results.json
│   ├── adaptation_extremes_test_results.json
│   ├── survival_scenario_test_results.json
│   └── protocol_v2_summary.json          # Aggregated summary
├── plots/
│   ├── baseline_comparison.png
│   ├── adaptation_extremes.png
│   ├── multimodality.png
│   └── [other visualizations]
└── reports/
    ├── summary_report.md                # Markdown report
    └── [PDF reports]                    # Future: publication-ready PDFs

```

## Key Metrics Tracked

### Per Test Suite

**Integration**: Component status, consolidation count, parameter changes, improvement %
**Usability**: API simplicity, error handling, configuration options, logging quality
**Baselines**: Loss curves, final loss, improvement %, comparison % vs base
**Multimodality**: Dimension, output shape, final loss, improvement % per modality
**Memory Stress**: Buffer size, consolidation count, memory MB, efficiency %, prioritization check
**Adaptation Extremes**: Task count, forgetting ratio, recovery rate %, stability %, adaptation quality %
**Survival**: Loss comparison, error count, recovery rate %, checkpoint integrity
**Visualization**: PNG plots and markdown reports for publication

## JSON Result Format

Each test suite produces a standardized JSON file:

```json
{
  "timestamp": "2024-01-20T15:30:00",
  "tests_passed": 6,
  "tests_failed": 0,
  "component_status": {
    "consciousness": "ACTIVE",
    "memory_handler": "HYBRID",
    "consolidation": "TRIGGERED"
  },
  "metrics": {
    "consolidations_triggered": 5,
    "prioritized_samples": 1240,
    "parameter_changes": 340,
    "training_improvement": 0.35
  },
  "detailed_results": [
    {"test_name": "test_1", "passed": true, "metrics": {...}},
    ...
  ]
}
```

## Expected Results

### ✓ All Tests Should Pass
- Integration: 6/6 ✓
- Usability: 6/6 ✓
- Baselines: 4/4 ✓ (with MirrorMind outperforming)
- Multimodality: 5/5 ✓
- Memory Stress: 5/5 ✓
- Adaptation Extremes: 4/4 ✓
- Survival: 4/4 ✓

### 📊 Expected Performance
- **Baseline Comparison**: MirrorMind v7.0 should show 15-30% improvement over base model
- **Memory Efficiency**: System should handle 10K samples with <500MB growth
- **Adaptation**: System should recover 70%+ from domain shifts
- **Robustness**: Error recovery should have 100% success rate

## Interpreting Results

### High Pass Rate (>95%)
✓ Framework is stable and well-integrated

### Lower Memory Stress Pass Rate
⚠ May need to tune consolidation frequency or replay buffer size

### Lower Adaptation Results
⚠ May need to adjust lambda values or enable stronger consciousness mode

### Lower Baseline Improvement
⚠ Check if consciousness and prioritized replay are enabled

## Publishing Guidelines

1. **Tables**: Use JSON results to generate metrics tables
2. **Figures**: Use PNG plots from `/plots/` directory
3. **Discussion**: Reference findings from `summary_report.md`
4. **Reproducibility**: Include command used and random seed
5. **Datasets**: Describe input dimensions and distribution

## Configuration Options

Each test suite can be customized:

```python
# Example: Custom configuration
tester = IntegrationTester(
    device='cuda',           # Use GPU
    seed=42,                 # Reproducible
    consciousness_enabled=True,
    memory_type='hybrid',
    prioritized_replay=True,
    adaptive_lambda=True
)
```

## Performance Notes

- **CPU Execution**: ~15-20 minutes for complete protocol
- **GPU Execution**: ~3-5 minutes (with CUDA-capable GPU)
- **Memory Usage**: ~1-2GB peak memory
- **Disk Space**: ~50MB for all results + visualizations

## Troubleshooting

### Test Fails to Run
```bash
# Ensure all dependencies installed
pip install torch numpy matplotlib seaborn psutil

# Verify MirrorMind is importable
python -c "from airbornehrs import AdaptiveFramework; print('OK')"
```

### Out of Memory
```python
# Reduce stress test size in test_memory_stress.py
LARGE_BUFFER_SIZE = 5000  # Reduce from 10000
```

### Visualizations Not Generated
```bash
# Check if result files exist
ls results/*.json

# Run visualization reporter separately
python visualization_reporter.py
```

## Next Steps

1. ✓ Run all tests with `python quick_start.py`
2. ✓ Review JSON results in `results/` directory
3. ✓ Examine plots in `plots/` directory
4. ✓ Read summary in `reports/summary_report.md`
5. ✓ Create publication figures from plots
6. ✓ Write methods section referencing Protocol v2

## Contributing

To add new tests:
1. Create `test_new_dimension.py` in `/tests/`
2. Follow standard pattern: class with `run_all()` method
3. Save results to JSON in `/results/`
4. Add to master runner in `run_protocol_v2.py`
5. Update visualization reporter for new plots

---

**Protocol v2 - MirrorMind v7.0 Validation Suite**

For paper submission: "Testing was conducted using Protocol v2, a comprehensive validation framework covering integration, usability, baseline comparisons, multi-modality support, memory stress, extreme adaptation, and survival scenarios."
