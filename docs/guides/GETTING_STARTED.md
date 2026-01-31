# 🧠 Getting Started with Airborne-Antara V2.0.0

Airborne-Antara "Synthetic Intuition" is designed to be the simplest way to add State-of-the-Art cognitive capabilities to any PyTorch model.

## 📦 1. Installation

```bash
pip install airborne-antara
```

## 🚀 2. The "3-Step" Cognitive Upgrade

You do not need to rewrite your model. Airborne-Antara wraps around it.

### Step 1: Import the Framework
```python
import torch
import torch.nn as nn
from airborne_antara import AdaptiveFramework, AdaptiveFrameworkConfig
```

### Step 2: Define Your Standard Model
This can be anything: a Transformer, a CNN, an LSTM, or a simple MLP.
```python
# Example: A simple perception network
my_model = nn.Sequential(
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 10) # 10 classes
)
```

### Step 3: Wrap & Initialize
Use the `production()` factory to automatically load the V2.0.0 SOTA defaults (World Model, MoE, Memory).
```python
# 1. Load SOTA Configuration
config = AdaptiveFrameworkConfig.production()

# 2. Inject Cognitive Features
# This wraps your model in a "Cognitive Shell" containing:
# - World Model (Foresight)
# - H-MoE (Sparse Computation)
# - Neural Health Monitor (Self-Repair)
agent = AdaptiveFramework(my_model, config)
```

## ⚡ 3. Training Loop

Replace your standard `loss = criterion(out, target); loss.backward()` with the agent's `train_step`.

```python
inputs = torch.randn(32, 64)
targets = torch.randint(0, 10, (32,))

# The agent handles:
# - Forward Pass (with internal prediction)
# - Loss Calculation (Task Loss + Surprise Loss + Sparsity Penalty)
# - Backpropagation
# - Self-Healing Checks
metrics = agent.train_step(inputs, target_data=targets)

print(f"Loss: {metrics['loss']:.4f}")
print(f"Predictive Surprise: {metrics.get('surprise', 0.0):.4f}")
print(f"Active Memory Nodes: {len(agent.memory.graph_memory.nodes)}")
```

## 🎮 4. Real-Time Dashboard (Interactive)

To see what your agent is "thinking" (its surprise levels, active experts, memory recall), run the dashboard in a separate terminal:

```bash
python -m airborne_antara --demo
```

## 🔧 Advanced Configuration

If you want manual control over the cognitive features:

```python
config = AdaptiveFrameworkConfig(
    # Core settings
    model_dim=64,
    learning_rate=1e-3,
    
    # Feature Toggles
    enable_world_model=True,    # Turn on/off Foresight
    use_hierarchical_moe=True,  # Turn on/off Sparse Experts
    use_graph_memory=True,      # Turn on/off Relational Memory
    enable_health_monitor=True, # Turn on/off Self-Repair
    
    # Hyperparameters
    world_model_loss_weight=0.1,
    num_moe_domains=2,
    num_moe_experts=4
)
```

## 📚 Next Steps
- **Technical Deep Dive**: Understand how the World Model works in [Synthetic Intuition Guide](../technical/SYNTHETIC_INTUITION.md).
- **API Reference**: See [full class documentation](../API_REFERENCE.md).
