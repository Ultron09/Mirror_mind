# 🧪 MirrorMind: Experimental Suite

This directory contains the complete verification suite for the MirrorMind framework. These experiments reproduce the core claims of  **Adaptive Meta-Learning** ,  **Test-Time Training (TTT)** , and  **Stabilizer Reflexes** .

## 🚀 Quick Run

Run any experiment module directly from the root directory:

```
python experiments/run_multi_model.py
```

## 📋 Complete Experiment Index

### 1. Universal Adaptation Benchmark (The "Omni-Test")

**Script:** `run_multi_model.py`

**Goal:** Prove domain-agnostic adaptation by switching data streams from **NLP** → **Vision** → **Audio** in a single training run.

**Key Result:** The "Stabilizer Reflex" (learning rate spike) should trigger exactly at domain boundaries, allowing the model to instantly reconfigure for new modalities.

**Output:** `universal_benchmark_results.png`

### 2. Production Simulation (The "Gauntlet")

**Script:** `run_production.py`

**Goal:** Verify "Lifelong Learning" capability. Deploy a "lazily trained" (underfitted) model and observe if it improves over time solely through exposure to live production data.

**Key Result:** Prediction error should trend downward over time, even with noise injection.

**Output:** `production_proof.png`

### 3. Extreme Stress Test (Multi-Stage Drift)

**Script:** `run_experiment.py`

**Goal:** Test the **Meta-Controller's** limit by cycling through radically different logical tasks: **Identity** → **Negation** → **Scramble** → **Chaos** (High Noise).

**Key Result:** High introspection uncertainty during the "Chaos" phase and successful adaptation to rule reversals.

**Output:** `proof_extreme.png`

### 4. The Evolution Gap (Curriculum Learning)

**Script:** `run_gap.py`

**Goal:** Bridge the reasoning capabilities between small and large models using  **Test-Time Training** . The model evolves arithmetic reasoning capabilities on the fly.

**Key Result:** Accuracy recovers and climbs after difficulty spikes, effectively "evolving" a solution.

**Output:** `evolution_gap_v2.png`

### 5. Operation Overdrive (Instant Memorization)

**Script:** `run_overdrive.py`

**Goal:** Demonstrate  **Instant Knowledge Acquisition** . The model must memorize a random password generated in the prompt and recall it immediately.

**Key Result:** Success rate should hit 100% immediately after the adaptation step, proving short-term weight memory.

**Output:** `overdrive_results.png`

### 6. ARC-AGI Benchmark

**Script:** `run_arc_gpt2.py`

**Goal:** Evaluate general intelligence using the  **Abstraction and Reasoning Corpus** . Uses an optimized `EnhancedLiveVisualizer` for real-time metrics.

**Key Result:** Improved few-shot performance on logical puzzles compared to a frozen baseline.

**Output:** `enhanced_dashboard.png` (Live visualization)

### 7. The Library of Babel (Live Demo)

**Script:** `run_war.py`

**Goal:** A real-time visualization of cognitive adaptation. The model reads a stream of text that shifts styles (Medical → Legal → Hacker).

**Key Result:** A live `matplotlib` window showing the Learning Rate spiking whenever the text style changes, visually demonstrating the "Surprise" and "Adaptation" reflex.

**Output:** Live GUI Window / Console Stream

### 8. MirrorMind Benchmark (Math & Code)

**Script:** `run_data.py`

**Goal:** A dual-task benchmark on **GSM8K** (Math) and **MBPP** (Python Code) datasets. Evaluates the framework's ability to switch between reasoning and coding.

**Key Result:** Comparative loss metrics between Base Model and MirrorMind across two distinct reasoning domains.

**Output:** `mirrormind_dashboard.png` and `mirrormind_results.csv`

## 🔧 Configuration

Experiments use the configuration files in `experiments/configs/`. You can customize hyperparameters (model size, learning rates) by modifying `sample_config.json` or passing arguments to specific runners if supported.

```
# Example (if supported by script args)
python experiments/run_experiment.py --config experiments/configs/sample_config.json
```
