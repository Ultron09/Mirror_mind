# MirrorMind API Reference (Framework v7.x)

This document provides API documentation for the core public classes and configuration objects in the MirrorMind framework.

---

## Table of Contents

1.  [Core Framework (`AdaptiveFramework`)](#1-core-framework-adaptiveframework)
2.  [Cognitive Components](#2-cognitive-components)
    -   [`EnhancedConsciousnessCore`](#21-enhancedconsciousnesscore)
    -   [`AdapterBank`](#22-adapterbank)
    -   [`UnifiedMemoryHandler`](#23-unifiedmemoryhandler)
    -   [`MetaController`](#24-metacontroller)
3.  [Configuration Objects](#3-configuration-objects)
    -   [`AdaptiveFrameworkConfig`](#31-adaptiveframeworkconfig)
    -   [`MetaControllerConfig`](#32-metacontrollerconfig)

---

## 1. Core Framework (`AdaptiveFramework`)

The `AdaptiveFramework` is the central wrapper that integrates all other components and orchestrates the learning process. You pass your `torch.nn.Module` to it, and it augments it with the four concurrent learning loops.

**Source:** `airbornehrs/core.py`

### Class Signature
```python
class AdaptiveFramework(nn.Module):
```

### Constructor

```python
def __init__(
    self,
    user_model: nn.Module,
    config: AdaptiveFrameworkConfig = None,
    device: str = None
):
```
| Parameter    | Type                  | Description                                                                  |
|--------------|-----------------------|------------------------------------------------------------------------------|
| `user_model` | `torch.nn.Module`     | The PyTorch model you want to make adaptive.                                 |
| `config`     | `AdaptiveFrameworkConfig` | Optional configuration dataclass. If `None`, default values are used.        |
| `device`     | `str`                 | Optional device string (e.g., `'cuda'`, `'cpu'`). Overrides `config.device`. |

### Key Methods

#### `train_step`
This is the main method used during training. It executes a single forward and backward pass, automatically handling all four learning loops (Task, Memory, Meta, Introspection).

```python
def train_step(
    self,
    input_data,
    target_data,
    enable_dream: bool = True,
    meta_step: bool = True,
    record_stats: bool = True
) -> Dict[str, Any]:
```
-   **Returns:** A dictionary containing metrics from the step, such as `loss`, `mse`, `z_score`, and `mode`.

#### `save_checkpoint`
Saves the state of the model and its components to a file.

```python
def save_checkpoint(self, path: str):
```

#### `load_checkpoint`
Loads the state of the model and its components from a file.

```python
def load_checkpoint(self, path: str):
```

---

## 2. Cognitive Components

These are the primary modules that are orchestrated by the `AdaptiveFramework`.

### 2.1. `EnhancedConsciousnessCore`

Provides high-level cognitive control, analyzing performance to generate a `learning_multiplier`.

**Source:** `airbornehrs/consciousness_v2.py`

#### Constructor
```python
def __init__(
    self,
    feature_dim: int = 256,
    awareness_buffer_size: int = 5000,
    novelty_threshold: float = 2.0,
    model: Optional[nn.Module] = None
):
```

#### `observe`
The core method that analyzes a single training step.

```python
def observe(
    self,
    x: torch.Tensor,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    task_id: str = "default",
    features: Optional[torch.Tensor] = None
) -> Dict[str, Any]:
```
-   **Returns:** A dictionary of metrics, including `emotion` and the crucial `learning_multiplier`.

### 2.2. `AdapterBank`

Manages parameter-efficient adapters that are injected into the user's model to enable plasticity without catastrophic forgetting.

**Source:** `airbornehrs/adapters.py`

#### Constructor
```python
def __init__(self, num_layers: int = 0, device: torch.device = None):
```

#### Key Methods

-   **`ensure_index(self, idx: int, out_dim: int = None)`**: Ensures an adapter exists for a given layer index, creating it if necessary.
-   **`apply(self, idx: int, activation: torch.Tensor)`**: Applies the appropriate adapter to a layer's activation.
-   **`parameters(self) -> Iterator[nn.Parameter]`**: Returns an iterator over all adapter parameters, for use in an optimizer.

### 23. `UnifiedMemoryHandler`

Prevents catastrophic forgetting by calculating and applying a penalty to the loss function for changes to important weights.

**Source:** `airbornehrs/memory.py`

#### Constructor
```python
def __init__(
    self,
    model: nn.Module,
    method: str = 'si',
    si_lambda: float = 1.0,
    si_xi: float = 1e-3,
    ewc_lambda: float = 0.4,
    consolidation_criterion: str = 'hybrid'
):
```
| Parameter    | Type   | Description                                     |
|--------------|--------|-------------------------------------------------|
| `model`      | `nn.Module`| The model to protect.                           |
| `method`     | `str`  | The protection method: `'si'`, `'ewc'`, or `'hybrid'`. |
| `si_lambda`  | `float`| Regularization strength for Synaptic Intelligence.|
| `ewc_lambda` | `float`| Regularization strength for EWC.                |

#### Key Methods

-   **`consolidate(self, ...)`**: Updates the parameter importance scores (Omega for SI, Fisher for EWC).
-   **`compute_penalty(self, ...)`**: Calculates the regularization loss to be added to the main task loss.
-   **`save_task_memory(self, ...)`**: Saves the current importance weights and anchors as a "task memory."
-   **`load_task_memory(self, ...)`**: Loads a previously saved task memory.

### 2.4. `MetaController`

Implements the Reptile meta-learning algorithm to help the model "learn to learn."

**Source:** `airbornehrs/meta_controller.py`

#### Constructor
```python
def __init__(
    self,
    framework: Any,
    config: Optional[MetaControllerConfig] = None
):
```

#### `adapt`
The main method called by the framework to perform a meta-update step.

```python
def adapt(
    self,
    loss: float,
    gradients: Optional[Dict[str, torch.Tensor]] = None,
    performance_metrics: Optional[Dict[str, float]] = None,
    external_grad_stats: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
```

---

## 3. Configuration Objects

These dataclasses are used to configure the framework and its components.

### 3.1. `AdaptiveFrameworkConfig`

**Source:** `airbornehrs/core.py`

```python
@dataclass
class AdaptiveFrameworkConfig:
    # Learning parameters
    learning_rate: float = 1e-3
    meta_learning_rate: float = 1e-4
    weight_adaptation_lr: float = 1e-5
    bias_adaptation_lr: float = 1e-5
    
    # Memory System
    memory_type: str = 'hybrid'  # 'ewc', 'si', or 'hybrid'
    consolidation_criterion: str = 'hybrid'
    consolidation_min_interval: int = 30
    consolidation_max_interval: int = 100
    consolidation_surprise_threshold: float = 2.5
    
    # Consciousness Layer
    enable_consciousness: bool = True
    consciousness_buffer_size: int = 5000
    novelty_threshold: float = 2.0
    
    # Optimization
    compile_model: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gradient_clip_norm: float = 1.0
    
    # ... and other parameters. See source for full list.
```

### 3.2. `MetaControllerConfig`

**Source:** `airbornehrs/meta_controller.py`

```python
@dataclass
class MetaControllerConfig:
    base_lr: float = 1e-3
    min_lr: float = 1e-6
    max_lr: float = 1e-2
    
    # Reptile Meta-Learning
    use_reptile: bool = True
    reptile_learning_rate: float = 0.1
    reptile_update_interval: int = 5
```
