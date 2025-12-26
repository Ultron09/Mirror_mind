# MirrorMind API Reference

## Complete API Documentation

This document provides comprehensive API documentation for all public classes, functions, and methods in the MirrorMind framework.

---

## Table of Contents

1. [Configuration](#1-configuration)
2. [Core Framework](#2-core-framework)
3. [Memory Handlers](#3-memory-handlers)
4. [Components](#4-components)
5. [Utilities](#5-utilities)

---

## 1. Configuration

### AdaptiveFrameworkConfig

```python
class AdaptiveFrameworkConfig(dataclass):
    """Configuration for the Universal Adaptive Framework."""
```

#### Architecture Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_dim` | int | 256 | Feature dimension for backbone |
| `num_layers` | int | 6 | Number of transformer layers |
| `num_heads` | int | 8 | Number of attention heads |
| `ff_dim` | int | 1024 | Feed-forward network dimension |
| `dropout` | float | 0.1 | Dropout rate (0.0-1.0) |

#### Learning Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | float | 1e-3 | Primary learning rate for model |
| `meta_learning_rate` | float | 1e-4 | Meta-learning rate for introspection |
| `weight_adaptation_lr` | float | 1e-5 | Learning rate for weight adapters |
| `bias_adaptation_lr` | float | 1e-5 | Learning rate for bias adapters |
| `adaptation_threshold` | float | 0.05 | Threshold for applying adaptations |

#### Memory Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `memory_type` | str | 'hybrid' | Memory mechanism: 'ewc', 'si', or 'hybrid' |
| `consolidation_criterion` | str | 'hybrid' | When to consolidate: 'time', 'surprise', or 'hybrid' |
| `consolidation_min_interval` | int | 50 | Minimum steps between consolidations |
| `consolidation_max_interval` | int | 500 | Maximum steps between consolidations |
| `consolidation_surprise_threshold` | float | 2.0 | Z-score threshold for surprise-based consolidation |

#### Consciousness Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_consciousness` | bool | False | Enable consciousness layer |
| `consciousness_buffer_size` | int | 5000 | Size of consciousness replay buffer |
| `novelty_threshold` | float | 2.0 | Z-score threshold for novelty detection |
| `use_attention` | bool | True | Enable attention mechanism |
| `use_intrinsic_motivation` | bool | True | Enable curiosity-driven learning |

#### Optimization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gradient_clip_norm` | float | 1.0 | Gradient clipping threshold |
| `adapter_max_norm` | float | 2.0 | Max norm for adapter parameters |
| `warmup_steps` | int | 50 | Steps for learning rate warmup |
| `use_amp` | bool | False | Enable automatic mixed precision |
| `compile_model` | bool | True | Compile model for speed |

#### Other Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | str | 'cuda' if available | Device: 'cpu', 'cuda', or 'mps' |
| `evaluation_frequency` | int | 10 | How often to evaluate (steps) |
| `feedback_buffer_size` | int | 10000 | Size of experience replay buffer |
| `dream_interval` | int | 10 | How often to run replay (steps) |
| `log_frequency` | int | 50 | How often to log (steps) |
| `checkpoint_frequency` | int | 500 | How often to checkpoint (steps) |

#### Example

```python
from airbornehrs import AdaptiveFrameworkConfig

config = AdaptiveFrameworkConfig(
    learning_rate=0.001,
    meta_learning_rate=0.0001,
    memory_type='ewc',
    enable_consciousness=True,
    device='cuda'
)
```

---

## 2. Core Framework

### AdaptiveFramework

```python
class AdaptiveFramework(nn.Module):
    """Universal Adaptive Meta-Learning Framework.
    
    Wraps any PyTorch model and adds:
    - Elastic Weight Consolidation for continual learning
    - Introspection RL for adaptive plasticity
    - Consciousness layer for self-awareness
    - Meta-learning for fast adaptation
    """
```

#### Constructor

```python
def __init__(
    self,
    user_model: nn.Module,
    config: AdaptiveFrameworkConfig = None,
    device: str = None
):
    """Initialize the AdaptiveFramework.
    
    Args:
        user_model: PyTorch model to wrap
        config: AdaptiveFrameworkConfig instance
        device: Device to use ('cpu', 'cuda', 'mps')
    
    Returns:
        AdaptiveFramework instance
    
    Raises:
        ValueError: If config is invalid
    
    Example:
        >>> model = MyNet()\n        >>> config = AdaptiveFrameworkConfig(learning_rate=0.001)\n        >>> framework = AdaptiveFramework(model, config)\n    \"\"\"\n```\n\n#### Forward Pass\n\n```python\ndef forward(self, x: torch.Tensor) -> torch.Tensor:\n    \"\"\"Forward pass with introspection.\n    \n    Args:\n        x: Input tensor (batch_size, ...)\n    \n    Returns:\n        Output tensor\n    \n    Example:\n        >>> x = torch.randn(32, 64)\n        >>> output = framework(x)\n    \"\"\"\n```\n\n#### Training Methods\n\n```python\ndef train(self, mode: bool = True) -> 'AdaptiveFramework':\n    \"\"\"Set training mode.\n    \n    Args:\n        mode: True for training, False for evaluation\n    \n    Returns:\n        Self\n    \"\"\"\n\ndef eval(self) -> 'AdaptiveFramework':\n    \"\"\"Set evaluation mode.\n    \n    Returns:\n        Self\n    \"\"\"\n```\n\n#### Memory Consolidation\n\n```python\ndef consolidate_memory(self, data: torch.Tensor = None) -> None:\n    \"\"\"Consolidate learned knowledge into long-term memory.\n    \n    Args:\n        data: Optional data to compute Fisher information on\n    \n    Note:\n        - Called automatically based on consolidation_criterion\n        - Can also be called manually after task completion\n    \n    Example:\n        >>> # After training on a task\n        >>> framework.consolidate_memory(X_task)\n    \"\"\"\n```\n\n#### Properties\n\n```python\n# Read-only properties\n@property\ndef model(self) -> nn.Module:\n    \"\"\"Get the underlying model.\"\"\"\n\n@property\ndef device(self) -> torch.device:\n    \"\"\"Get the device the framework is on.\"\"\"\n\n@property\ndef is_training(self) -> bool:\n    \"\"\"Check if in training mode.\"\"\"\n```\n\n#### Example Usage\n\n```python\nimport torch\nimport torch.nn as nn\nfrom airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig\n\n# Create your model\nclass MyModel(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc1 = nn.Linear(64, 128)\n        self.fc2 = nn.Linear(128, 10)\n    \n    def forward(self, x):\n        x = torch.relu(self.fc1(x))\n        return self.fc2(x)\n\n# Wrap with MirrorMind\nmodel = MyModel()\nconfig = AdaptiveFrameworkConfig(\n    learning_rate=0.001,\n    enable_consciousness=True\n)\nframework = AdaptiveFramework(model, config, device='cuda')\n\n# Training loop\nfor epoch in range(10):\n    for X_batch, y_batch in train_loader:\n        framework.train()\n        \n        # Forward pass\n        output = framework(X_batch)\n        loss = criterion(output, y_batch)\n        \n        # Backward pass\n        framework.optimizer.zero_grad()\n        loss.backward()\n        framework.optimizer.step()\n\n# Evaluate\nframework.eval()\nwith torch.no_grad():\n    output = framework(X_test)\n    predictions = output.argmax(dim=1)\n\n# Consolidate after learning\nframework.consolidate_memory(X_train)\n```\n\n---\n\n## 3. Memory Handlers\n\n### EWCHandler\n\n```python\nclass EWCHandler:\n    \"\"\"Elastic Weight Consolidation for continual learning.\n    \n    Implements EWC as described in:\n    'Overcoming catastrophic forgetting in neural networks' (Kirkpatrick et al.)\n    \"\"\"\n```\n\n#### Methods\n\n```python\ndef compute_fisher(\n    self,\n    data_loader: DataLoader,\n    num_batches: int = None\n) -> None:\n    \"\"\"Compute Fisher information matrix.\n    \n    Args:\n        data_loader: DataLoader with training data\n        num_batches: Number of batches to process (None = all)\n    \n    Note:\n        - Stores task parameters for later use\n        - Called during consolidation\n    \"\"\"\n\ndef regularization_loss(self) -> torch.Tensor:\n    \"\"\"Compute EWC regularization penalty.\n    \n    Returns:\n        Scalar regularization loss\n    \n    Note:\n        - Added to task loss during training\n        - Prevents forgetting of previous tasks\n    \"\"\"\n```\n\n### SIHandler\n\n```python\nclass SIHandler:\n    \"\"\"Synaptic Intelligence memory handler.\n    \n    Tracks importance of parameters online during training.\n    \"\"\"\n```\n\n#### Methods\n\n```python\ndef update_importances(self) -> None:\n    \"\"\"Update parameter importance scores.\n    \n    Note:\n        - Called automatically during training\n        - Tracks parameter changes and gradients\n    \"\"\"\n\ndef regularization_loss(self) -> torch.Tensor:\n    \"\"\"Compute SI regularization penalty.\n    \n    Returns:\n        Scalar regularization loss\n    \"\"\"\n```\n\n---\n\n## 4. Components\n\n### ConfigValidator\n\n```python\nclass ConfigValidator:\n    \"\"\"Validates AdaptiveFrameworkConfig instances.\n    \n    Checks 30+ configuration parameters for correctness.\n    \"\"\"\n```\n\n#### Methods\n\n```python\ndef validate(self, config: AdaptiveFrameworkConfig) -> Tuple[bool, List[str], List[str]]:\n    \"\"\"Validate configuration.\n    \n    Args:\n        config: Configuration to validate\n    \n    Returns:\n        (is_valid, errors, warnings)\n        - is_valid: True if configuration is valid\n        - errors: List of error messages (blocking)\n        - warnings: List of warning messages (advisory)\n    \n    Example:\n        >>> validator = ConfigValidator()\n        >>> is_valid, errors, warnings = validator.validate(config)\n        >>> if not is_valid:\n        ...     for error in errors:\n        ...         print(f\"Error: {error}\")\n    \"\"\"\n\ndef print_report(self, is_valid: bool, errors: List[str], warnings: List[str]) -> None:\n    \"\"\"Print formatted validation report.\n    \n    Args:\n        is_valid: Whether validation passed\n        errors: List of errors\n        warnings: List of warnings\n    \"\"\"\n```\n\n#### Function\n\n```python\ndef validate_config(\n    config: AdaptiveFrameworkConfig,\n    raise_on_error: bool = True\n) -> Union[bool, Tuple[bool, List[str], List[str]]]:\n    \"\"\"Convenience function to validate config.\n    \n    Args:\n        config: Configuration to validate\n        raise_on_error: If True, raise ValueError on errors\n    \n    Returns:\n        - If raise_on_error=True: bool (valid or raises)\n        - If raise_on_error=False: (valid, errors, warnings) tuple\n    \n    Example:\n        >>> config = AdaptiveFrameworkConfig()\n        >>> is_valid, errors, warnings = validate_config(config, raise_on_error=False)\n    \"\"\"\n```\n\n### FeedbackBuffer\n\n```python\nclass FeedbackBuffer:\n    \"\"\"Experience replay buffer for continual learning.\n    \n    Uses reservoir sampling for efficient buffer management.\n    \"\"\"\n```\n\n#### Methods\n\n```python\ndef add(\n    self,\n    input_data: torch.Tensor,\n    output: torch.Tensor,\n    target: torch.Tensor,\n    reward: float,\n    loss: float\n) -> None:\n    \"\"\"Add experience to buffer.\n    \n    Args:\n        input_data: Model input\n        output: Model output\n        target: Ground truth target\n        reward: Reward signal\n        loss: Loss value\n    \"\"\"\n\ndef sample(\n    self,\n    batch_size: int,\n    device: str = 'cpu'\n) -> List[torch.Tensor]:\n    \"\"\"Sample random batch from buffer.\n    \n    Args:\n        batch_size: Number of samples to return\n        device: Device to move tensors to\n    \n    Returns:\n        [X_batch, y_batch, loss_batch]\n    \"\"\"\n\ndef clear(self) -> None:\n    \"\"\"Clear all samples from buffer.\"\"\"\n```\n\n### MetaController\n\n```python\nclass MetaController:\n    \"\"\"Meta-learning controller for adaptive updates.\n    \n    Implements Reptile algorithm for fast-weight updates.\n    \"\"\"\n```\n\n#### Methods\n\n```python\ndef update(self) -> None:\n    \"\"\"Perform meta-learning update.\n    \n    Note:\n        - Updates fast weights based on task performance\n        - Called during training\n    \"\"\"\n\ndef get_fast_weights(self) -> Dict[str, torch.Tensor]:\n    \"\"\"Get fast-adapted weights.\n    \n    Returns:\n        Dictionary of adapted parameter names to values\n    \"\"\"\n```\n\n### Consciousness Layer\n\n```python\nclass ConsciousnessCore:\n    \"\"\"Self-aware learning component.\n    \n    Monitors model's own uncertainty and knowledge gaps.\n    \"\"\"\n```\n\n#### Methods\n\n```python\ndef update(self, x: torch.Tensor, logits: torch.Tensor) -> None:\n    \"\"\"Update consciousness based on input and predictions.\n    \n    Args:\n        x: Input tensor\n        logits: Model output logits\n    \"\"\"\n\ndef get_awareness_signal(self) -> torch.Tensor:\n    \"\"\"Get current awareness level.\n    \n    Returns:\n        Scalar tensor in range [0, 1]\n    \"\"\"\n```\n\n### Attention Mechanism\n\n```python\nclass AttentionMechanism:\n    \"\"\"Learn feature importance weights.\n    \n    Automatically identifies which features matter most.\n    \"\"\"\n```\n\n#### Methods\n\n```python\ndef forward(self, x: torch.Tensor) -> torch.Tensor:\n    \"\"\"Compute attention weights.\n    \n    Args:\n        x: Input features\n    \n    Returns:\n        Attention weights (same shape as input)\n    \"\"\"\n```\n\n---\n\n## 5. Utilities\n\n### Model Checkpointing\n\n```python\ndef save_checkpoint(\n    framework: AdaptiveFramework,\n    path: str,\n    task_name: str = None\n) -> None:\n    \"\"\"Save framework checkpoint.\n    \n    Args:\n        framework: AdaptiveFramework to save\n        path: Path to save checkpoint\n        task_name: Optional task identifier\n    \"\"\"\n\ndef load_checkpoint(\n    framework: AdaptiveFramework,\n    path: str\n) -> AdaptiveFramework:\n    \"\"\"Load framework checkpoint.\n    \n    Args:\n        framework: AdaptiveFramework to load into\n        path: Path to checkpoint file\n    \n    Returns:\n        Loaded framework\n    \"\"\"\n```\n\n### Performance Utilities\n\n```python\ndef compute_metrics(\n    predictions: torch.Tensor,\n    targets: torch.Tensor,\n    task_name: str = None\n) -> Dict[str, float]:\n    \"\"\"Compute performance metrics.\n    \n    Args:\n        predictions: Model predictions\n        targets: Ground truth labels\n        task_name: Optional task identifier\n    \n    Returns:\n        Dictionary with metrics: {'accuracy', 'loss', 'confusion_matrix'}\n    \"\"\"\n```\n\n---\n\n## Summary\n\n### Quick Reference\n\n```python\n# Import\nfrom airbornehrs import (\n    AdaptiveFramework,\n    AdaptiveFrameworkConfig,\n    ConfigValidator,\n    validate_config\n)\n\n# Create and use\nconfig = AdaptiveFrameworkConfig(learning_rate=0.001)\nframework = AdaptiveFramework(your_model, config)\n\n# Validate\nis_valid, errors, warnings = validate_config(config, raise_on_error=False)\n\n# Train\nfor X_batch, y_batch in data_loader:\n    output = framework(X_batch)\n    loss = criterion(output, y_batch)\n    framework.optimizer.zero_grad()\n    loss.backward()\n    framework.optimizer.step()\n\n# Consolidate\nframework.consolidate_memory(X_train)\n\n# Evaluate\nframework.eval()\nwith torch.no_grad():\n    output = framework(X_test)\n```\n\n---\n\n*API Documentation for MirrorMind v6.5*\n