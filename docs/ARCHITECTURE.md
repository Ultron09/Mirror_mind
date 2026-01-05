# MirrorMind Framework Architecture (v1.1.1 "Sentient" Edition)

## 1. High-Level Overview

The MirrorMind framework is designed to wrap any standard `torch.nn.Module` and augment it with the capabilities for stable and continuous learning. The core design philosophy is to move beyond static, train-then-deploy models and towards "living systems" that can adapt safely in non-stationary environments.

This is achieved through the integration of **four concurrent control loops** that operate during each training step. Each loop is responsible for a different aspect of adaptive intelligence, and they work together to produce a single, robust weight update.

The four loops are:
1.  **The Task Loop:** Standard gradient descent to learn the current task.
2.  **The Memory Loop:** Prevents catastrophic forgetting of past knowledge.
3.  **The Meta Loop:** Enables the model to "learn to learn" and adapt more quickly.
4.  **The Introspection Loop:** Provides high-level cognitive control and dynamic plasticity.
5.  **The System 2 Loop (V8.0):** Recursive Global Workspace for multi-step reasoning.

![High-Level Diagram](https://i.imgur.com/your-diagram-url.png)
*(Note: A proper diagram should be created and linked here)*

---

## 2. The Four Loops in Detail

### 2.1. The Task Loop: Learning the 'Now'

-   **Purpose:** To learn the immediate task at hand.
-   **Components:** The user's base `nn.Module`, standard loss functions (e.g., `CrossEntropyLoss`), and an optimizer (`AdamW`).
-   **Mechanism:** This is the standard deep learning forward and backward pass. The framework calculates the loss for the current batch and computes the gradients with respect to the model's parameters. This loop answers the question: "How should I change to get better at this specific data?"

### 2.2. The Memory Loop: Protecting the 'Past'

-   **Purpose:** To prevent catastrophic forgetting—the tendency of neural networks to discard old knowledge when learning new tasks.
-   **Component:** `UnifiedMemoryHandler` (`airbornehrs/memory.py`)
-   **Mechanism:** The `UnifiedMemoryHandler` adds a penalty term to the main task loss. This penalty makes it "harder" for the optimizer to change weights that were important for previous tasks.
    -   It uses a **hybrid approach**, combining two well-established techniques:
        1.  **Elastic Weight Consolidation (EWC):** Calculates the **Fisher Information Matrix** to estimate how important each weight is to a task. Changes to important weights are heavily penalized.
        2.  **Synaptic Intelligence (SI):** Calculates parameter importance online by accumulating how much each parameter contributes to loss changes over its trajectory. It is less computationally expensive than EWC.
    -   After a task is complete (or when triggered by surprise), the framework calls `memory.consolidate()` to compute and "lock in" the importance of the weights for that task.

### 2.3. The Meta Loop: Learning to Learn for the 'Future'

-   **Purpose:** To enable the model to adapt more quickly and efficiently to new tasks or distributions in the future.
-   **Component:** `MetaController` (`airbornehrs/meta_controller.py`)
-   **Mechanism:** The `MetaController` implements **Reptile**, a first-order meta-learning algorithm.
    -   It maintains two sets of weights: **slow weights** (for long-term, general knowledge) and **fast weights** (the model's active weights for short-term adaptation).
    -   Periodically, the slow weights are updated by interpolating them towards the fast weights: `slow_weights = slow_weights + epsilon * (fast_weights - slow_weights)`.
    -   This process trains the model's initial parameters (`slow_weights`) to be a good starting point for fine-tuning on a wide range of tasks, effectively learning a good "general initialization."

### 2.4. The Introspection Loop: Cognitive Control & Plasticity

-   **Purpose:** To provide high-level, strategic control over the learning process and to dynamically adjust the model's architecture for the task at hand.
-   **Components:** `EnhancedConsciousnessCore` (`airbornehrs/consciousness_v2.py`) and `AdapterBank` (`airbornehrs/adapters.py`).
-   **Mechanism:** This is the most sophisticated loop, acting as the framework's "brain."
    1.  **Observation:** The `EnhancedConsciousnessCore` observes the model's performance on the current batch, calculating metrics like `surprise` (how unexpected the error is), `uncertainty`, and `confidence`.
    2.  **Emotional Modulation:** It maps these metrics to a simulated "emotional state" (e.g., `CURIOUS`, `BORED`, `FRUSTRATED`).
    3.  **Learning Rate Adjustment:** Based on the emotion, it produces a `learning_multiplier` that is applied to the task loss. For example, if the model is 'frustrated' (high error, low progress), it might increase the learning rate to try and break out of a local minimum. If 'bored' (low error, low novelty), it might decrease it to save computational effort.
    4.  **Dynamic Plasticity:** The `AdapterBank` injects small, parameter-efficient "adapter" modules (like FiLM or Bottleneck layers) into the base model. The Introspection Loop can decide which adapters to activate for a given task, allowing the model to learn task-specific knowledge in these adapters while leaving the core pre-trained weights largely untouched. This is a key mechanism for both preventing forgetting and enabling multi-task learning.

### 2.5. The System 2 Loop (V8.0): Recursive Reasoning

-   **Purpose:** To enable multi-step "thinking" for complex problems that require more than a single forward pass.
-   **Component:** `RecursiveGlobalWorkspace` in `ConsciousnessCore` (`airbornehrs/consciousness_v2.py`)
-   **Mechanism:**
    1.  **Perception:** Input features are projected into a global workspace.
    2.  **Recursive Attention:** Multi-head attention is applied iteratively, allowing the model to "think" about the problem.
    3.  **Thought Traces:** Each iteration is logged to `current_thought_trace`, enabling introspection of the reasoning process.
    4.  **Adaptive Depth:** The number of iterations can adapt based on problem complexity (confusion level).

---

## 3. Data Flow in a Single `train_step`

Here’s how these loops interact during a single call to `framework.train_step(input, target)`:

1.  **Forward Pass:** The input data goes through the `AdapterBank`-modified model. The `EnhancedConsciousnessCore` observes the layer activations and final output.
2.  **Loss Calculation:**
    a. The **Task Loop** calculates the primary loss (e.g., cross-entropy) based on the model's output and the target.
    b. The `EnhancedConsciousnessCore` (**Introspection Loop**) calculates its internal metrics and produces a `learning_multiplier`.
    c. The task loss is multiplied by the `learning_multiplier`.
    d. The `UnifiedMemoryHandler` (**Memory Loop**) calculates a penalty term based on the importance of the weights being changed.
    e. **Total Loss = (Task Loss * Learning Multiplier) + Memory Penalty**.
3.  **Backward Pass:** Gradients are computed based on this `Total Loss`.
4.  **Optimizer Step:** The `AdamW` optimizer updates the model's weights (the "fast weights").
5.  **Meta-Update:** The `MetaController` (**Meta Loop**) periodically updates the "slow weights" based on the trajectory of the fast weights.

This unified process ensures that every single weight update is informed not just by the immediate task, but also by the constraints of past knowledge, future adaptability, and the model's current cognitive state.
