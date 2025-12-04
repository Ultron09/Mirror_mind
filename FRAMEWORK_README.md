# MirrorMind: Self-Learning Research Framework

> **A reproducible research platform for adaptive meta-learning and model-level introspection**

## 🎯 What is MirrorMind?

MirrorMind is a research-oriented self-learning framework where:

- **The Model** = "The Gun" (learns from experience)
- **The Framework** = "The Stabilizer" (adapts training dynamics to improve learning)
- **The Loop** = Iterative improvement through feedback and adaptation

Think of it as a meta-learning platform: the model learns tasks and the framework provides adaptive guidance to improve learning.

### Key Innovation

Instead of just training a model, MirrorMind creates a **feedback loop** where:
1. Model makes predictions
2. Gets feedback from environment
3. Analyzes its own performance
4. Adjusts its weights and learning strategy
5. Repeats → Continuous improvement

---

## 🚀 Quick Start

### Installation

```bash
# Clone or download MirrorMind
cd MirrorMind

# Install dependencies
pip install torch numpy tqdm
```

### 30-Second Example

```python
from AGITrainer import EasyTrainer

# Create trainer (just 1 line!)
trainer = EasyTrainer()

# Your data
X = torch.randn(1000, 10, 128)  # 1000 samples, seq_len=10, dim=128
y = torch.randn(1000, 10, 128)

# Train (automatically handles self-improvement!)
trainer.train(X, y, epochs=10)

# Predict
predictions = trainer.predict(X_test)

# That's it! 🎉
```

---

## 📦 Architecture Overview

```
MirrorMind Framework
├── SelfLearningFramework.py       ← Core self-learning engine
├── AdvancedAdaptation.py          ← Meta-learning & adaptation
├── AGITrainer.py                  ← Training orchestrator
├── Dashboard.py                   ← Monitoring & visualization
└── README.md                      ← This file
```

### Component Breakdown

#### 1. **SelfLearningFramework** - The Foundation
Core engine that manages:
- Model architecture with introspection
- Weight adaptation based on performance
- Feedback collection
- Learning state tracking

```python
framework = SelfLearningFramework(config)
output = framework.forward(x)
metrics = framework.train_step(x, y)
```

#### 2. **AdvancedAdaptation** - The Intelligence
Advanced learning mechanisms:
- **MAMLAdaptor** - Learn to learn (MAML-style meta-learning)
- **GradientAnalyzer** - Understand learning dynamics
- **DynamicLearningRateScheduler** - Adapt learning rate on-the-fly
- **CurriculumLearning** - Start easy, get harder
- **PerformanceAnalyzer** - Detect overfitting/underfitting
- **AdaptiveTaskSelector** - Focus on hard tasks

```python
# Automatic curriculum learning
curriculum = CurriculumLearningStrategy(config)
difficulty = curriculum.sample_task_difficulty()

# Dynamic learning rate
scheduler = DynamicLearningRateScheduler(optimizer, config, lr)
scheduler.step(loss)
```

#### 3. **AGITrainer** - The Orchestrator
Complete training pipeline:
- Manages training loop
- Coordinates all components
- Handles checkpointing
- Generates reports

```python
trainer = AGITrainer(config, environment)
summary = trainer.train(train_loader, val_loader, epochs=10)
```

#### 4. **EasyTrainer** - Simple API
Ultra-simple interface:
```python
trainer = EasyTrainer()
trainer.train(X, y, epochs=10)
predictions = trainer.predict(X_test)
```

#### 5. **Dashboard** - Monitoring
Real-time visualization:
- Metrics tracking
- HTML dashboard
- Performance reports

---

## 🔄 The Self-Learning Loop

Here's how MirrorMind creates continuous improvement:

```
1. FORWARD PASS
   ├─ Model processes input
   ├─ Internal states captured
   └─ Output generated

2. EVALUATION
   ├─ Compare with target
   ├─ Compute loss
   └─ Get reward signal

3. ANALYSIS (Self-Introspection)
   ├─ Analyze own attention patterns
   ├─ Check gradient health
   ├─ Assess learning efficiency
   └─ Detect overfitting/underfitting

4. ADAPTATION (Self-Improvement)
   ├─ Adjust weights based on analysis
   ├─ Modify learning rate dynamically
   ├─ Update learning strategy
   └─ Plan next training approach

5. FEEDBACK (Learning)
   ├─ Store experience
   ├─ Learn from mistakes
   └─ Improve predictions

6. REPEAT → Continuous Improvement! ↩
```

---

## 💡 Key Features

### Meta-Cognitive Attention (research probe)
Components in the codebase provide attention diagnostics intended for
controlled experiments. These diagnostics can be used to analyze attention
statistics and to inform adaptive mechanisms during training. They are
research probes and should be evaluated via ablation studies and baselines.
- Analyze attention entropy and related statistics
- Produce attention-quality diagnostics for downstream modules
- Optionally drive adaptive interventions based on diagnostics

### 🎓 Meta-Learning (MAML-style)
Learn how to learn:
- Inner loop: Adapt to tasks
- Outer loop: Improve adaptation strategy
- Few-shot learning capability

### 📊 Dynamic Learning
Adaptive mechanisms that respond to learning progress:
- Learning rate scheduling
- Curriculum learning (easy → hard)
- Gradient monitoring
- Performance-based adaptation

### Self-Monitoring and Introspection (research probe)
The framework provides utilities for internal monitoring (confidence
calibration, uncertainty estimation, and reasoning traces). These are
algorithmic tools for analyzing model behavior and are not evidence of
consciousness or sentience. Use them for measurement and controlled
experiments only.
- Confidence calibration and uncertainty diagnostics
- Internal consistency and coherence metrics
- Reasoning trace extraction for interpretability studies

### 💾 Experience Management
Smart experience handling:
- Episodic memory with importance weighting
- Curriculum-based task selection
- Adaptive task difficulty
- Experience replay

---

## 📈 Usage Examples

### Example 1: Simple Regression

```python
from AGITrainer import EasyTrainer
import torch

# Create trainer
trainer = EasyTrainer()

# Generate data
X = torch.randn(100, 10, 128)
y = torch.randn(100, 10, 128)

# Train with automatic self-improvement
summary = trainer.train(X, y, epochs=5)

# Results
print(f"Best loss: {summary['best_train_loss']:.4f}")
print(f"Epochs: {summary['epochs_completed']}")

# Make predictions
predictions = trainer.predict(X[:10])
print(f"Predictions shape: {predictions.shape}")
```

### Example 2: Custom Environment

```python
from AGITrainer import Environment, AGITrainer

class MyEnvironment(Environment):
    def reset(self):
        super().reset()
        # Your reset logic
    
    def step(self, action):
        # Your environment logic
        observation = ...
        reward = ...
        done = ...
        return observation, reward, done

# Use with trainer
env = MyEnvironment()
trainer = AGITrainer(config, environment=env)
summary = trainer.train(train_loader, num_epochs=20)
```

### Example 3: Advanced Configuration

```python
from SelfLearningFramework import FrameworkConfig
from AGITrainer import AGITrainer

# Custom config
config = FrameworkConfig(
    model_dim=256,              # Model dimension
    num_layers=8,               # Number of transformer layers
    num_heads=8,                # Attention heads
    learning_rate=1e-3,         # Base learning rate
    meta_learning_rate=1e-4,    # Meta-learning rate
    batch_size=32,
    epochs=20,
    weight_adaptation_lr=1e-5,  # How aggressively to adapt weights
    evaluation_frequency=100,   # Evaluate every N steps
    adaptation_threshold=0.05   # Adapt if improvement < 5%
)

# Create trainer
trainer = AGITrainer(config)
```

### Example 4: Monitoring Training

```python
from Dashboard import MetricsTracker, DashboardGenerator

# Create tracker
tracker = MetricsTracker()

# During training, add metrics
for epoch in range(10):
    train_loss = ...
    val_loss = ...
    tracker.add_metric(
        training_loss=train_loss,
        validation_loss=val_loss,
        learning_rate=current_lr,
        efficiency=compute_efficiency(),
        step=epoch
    )

# Generate dashboard
dashboard = DashboardGenerator(tracker)
html = dashboard.generate_html()

# Save and open in browser
with open("dashboard.html", "w") as f:
    f.write(html)
```

---

## 🎯 How It Works: Under the Hood

### 1. Self-Learning Model
```python
# Model with introspection capability
class SelfLearningModel(nn.Module):
    def forward(self, x, return_internals=False):
        # ... forward pass ...
        if return_internals:
            return output, uncertainty, internal_states
```

### 2. Adaptive Weight Manager
```python
# Manages weight adaptation
adapter = AdaptiveWeightManager(model, config, device)

# Analyze performance
importance = adapter.compute_layer_importance(activations)

# Adapt weights based on learning progress
adaptation_mag, _ = adapter.adapt_weights(
    current_loss, previous_loss, internals
)
```

### 3. Dynamic Learning Rate
```python
# Automatically adjusts learning rate
scheduler = DynamicLearningRateScheduler(optimizer, config, lr)

for epoch in epochs:
    for batch in data:
        loss = compute_loss(batch)
        scheduler.step(loss)  # Adjusts LR based on progress
```

### 4. Curriculum Learning
```python
# Gradually increase difficulty
curriculum = CurriculumLearningStrategy(config)

for task in tasks:
    difficulty = curriculum.sample_task_difficulty()
    success = train_on_task(task, difficulty)
    curriculum.report_task_completion(success)
```

---

## 📊 Monitoring & Visualization

MirrorMind includes built-in monitoring:

### Metrics Dashboard
```python
from Dashboard import MetricsTracker, DashboardGenerator

tracker = MetricsTracker()
# ... add metrics during training ...
dashboard = DashboardGenerator(tracker)
html = dashboard.generate_html()
```

### Console Logging
Automatic logging of:
- Loss trajectories
- Adaptation events
- Learning efficiency
- Gradient health
- Training progress

### Report Generation
```python
from Dashboard import ReportGenerator

ReportGenerator.generate_markdown_report(tracker, "report.md")
```

---

## 🔧 Configuration Guide

### Key Hyperparameters

```python
FrameworkConfig(
    # Architecture
    model_dim=256,                    # Model dimension
    num_layers=6,                     # Transformer layers
    num_heads=8,                      # Attention heads
    ff_dim=1024,                      # Feedforward dimension
    
    # Learning
    learning_rate=1e-3,               # Base learning rate
    meta_learning_rate=1e-4,          # Meta-learning rate
    batch_size=32,
    epochs=10,
    
    # Adaptation
    weight_adaptation_lr=1e-5,        # Weight adjustment speed
    bias_adaptation_lr=1e-5,
    evaluation_frequency=100,         # When to evaluate/adapt
    adaptation_threshold=0.05,        # Threshold for adaptation
    
    # Meta-learning
    inner_loop_steps=5,               # MAML inner loop
    outer_loop_steps=1,
    
    # Logging
    log_frequency=50,                 # Log every N steps
    checkpoint_frequency=500
)
```

### When to Adjust

- **High loss**: Increase `learning_rate` or reduce `batch_size`
- **Slow convergence**: Increase `weight_adaptation_lr`
- **Overfitting**: Enable curriculum learning or reduce `model_dim`
- **GPU memory**: Reduce `batch_size` or `model_dim`
- **Too slow**: Reduce `inner_loop_steps` or `num_layers`

---

## 📚 Core Concepts

### Self-Learning Loop
The framework continuously:
1. **Learns** from feedback
2. **Analyzes** its performance
3. **Adapts** its strategy
4. **Improves** over time

### Meta-Learning
The model learns not just to solve tasks, but to:
- Adapt quickly to new tasks
- Adjust learning strategy
- Optimize its own weights
- Improve learning efficiency

### Curriculum Learning
Training follows a natural progression:
- Start with easy tasks
- Gradually increase difficulty
- Model develops robust foundations
- Able to tackle hard problems

### Adaptive Components
Everything adapts based on performance:
- Learning rate changes based on loss
- Task difficulty increases gradually
- Weight updates scale with importance
- Adaptation threshold responds to progress

---

## 🎓 Research & Publications

This framework is suitable for research on:
- Meta-learning in neural networks
- Self-improving AI systems
- Curriculum learning
- Adaptive learning rate scheduling
- Model introspection and self-awareness
- AGI-oriented architectures

Potential papers:
- "MirrorMind: Self-Learning Neural Networks via Adaptive Meta-Learning"
- "Meta-Cognitive Attention for Continuous Self-Improvement"
- "Curriculum Learning with Dynamic Difficulty Adaptation"

---

## 🐛 Troubleshooting

### Training Diverges
- Reduce `learning_rate`
- Check for exploding gradients
- Reduce batch size
- Use gradient clipping (already enabled)

### Training Stuck (Plateaus)
- Increase `weight_adaptation_lr`
- Enable curriculum learning
- Reduce learning rate with scheduler
- Check for overfitting

### Slow Training
- Reduce `model_dim` or `num_layers`
- Increase `batch_size`
- Use GPU (set appropriate device)
- Reduce `inner_loop_steps`

### Out of Memory
- Reduce `batch_size`
- Reduce `model_dim`
- Reduce `num_layers`
- Use gradient accumulation

---

## 📝 License

MIT License - Use freely for research and commercial purposes

---

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- New adaptation strategies
- Additional meta-learning algorithms
- Improved visualization
- Domain-specific implementations
- Performance optimizations

---

## 🚀 Next Steps

1. **Start Simple**: Try the EasyTrainer with your data
2. **Monitor**: Use Dashboard for real-time feedback
3. **Tune**: Adjust config for your specific problem
4. **Deploy**: Use trained model for predictions
5. **Research**: Extend for novel architectures

---

## 📞 Support

For issues or questions:
1. Check the examples above
2. Review configuration guide
3. Check console logs for detailed error messages
4. Refer to component documentation in source code

---

## 🎉 Let's Build AGI Together!

MirrorMind represents a new paradigm in AI development:
- **Self-improving** systems
- **Adaptive** learning strategies
- **Introspective** models
- **Continuous** enhancement

**The model learns, and the framework ensures it learns better.**

Happy learning! 🧠✨
