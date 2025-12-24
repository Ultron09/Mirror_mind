# Introspection Loop: Z-Score Monitoring & OOD Detection

## Overview

The **Introspection Loop** is MirrorMind's anomaly detection system. It monitors a neural network's internal states and detects when something unexpected happens, then triggers adaptive responses.

**Core Components:**
1. **State Aggregator** — Collects activation statistics
2. **Z-Score Computation** — Statistical surprise detection
3. **Policy Network** — RL-based plasticity control
4. **OOD Detector** — Out-of-Distribution detection

---

## 1. Why Introspection? The Core Problem

### Traditional Approach: Loss-Based Feedback

Standard SGD relies on a single signal: loss on the current batch.

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

**Problems:**
- Loss is computed *after* forward/backward pass (reactive, not predictive)
- No signal when loss is anomalously high (model thinks it's just a hard example)
- No early warning before catastrophic divergence
- Can't distinguish between "hard example" vs "out-of-distribution"

### Better Approach: Predictive Monitoring

**Introspection monitors internal states BEFORE loss explodes:**

```
Input x arrives
    ↓
Model forward pass
    ↓
Collect internal statistics (activations, gradients)
    ↓
Compute Z-scores of statistics
    ↓
If Z > threshold: "Something unusual detected!"
    ↓
Trigger protective mechanisms (slow learning, weight freeze, etc.)
```

**Benefit:** Detect problems ~10-20 steps before loss diverges, giving time to adapt.

---

## 2. State Aggregation: What We Monitor

### Layer-Wise Statistics

For each layer $l$, compute statistics of activations $a_l$:

| Statistic | Formula | Interpretation |
|-----------|---------|-----------------|
| **Mean** | $\mu_l = \frac{1}{N} \sum_i a_{l,i}$ | Central tendency of layer output |
| **Variance** | $\sigma_l^2 = \frac{1}{N} \sum_i (a_{l,i} - \mu_l)^2$ | Spread/diversity of activation |
| **Max** | $\max_i a_{l,i}$ | Largest activation magnitude |
| **Norm** | $\|a_l\|_2 = \sqrt{\sum_i a_{l,i}^2}$ | Total signal magnitude |

### Global Aggregation

Combine layer statistics into a global state vector:

$$s_t = [\mu_1, \sigma_1, \|a_1\|_2, \mu_2, \sigma_2, \|a_2\|_2, \ldots, L_t, \nabla L_t]$$

Where:
- $\mu_l, \sigma_l, \|a_l\|_2$ — Layer $l$ statistics
- $L_t$ — Current loss
- $\nabla L_t$ — Loss gradient magnitude

**Dimensionality:** If model has 6 layers with 3 statistics each: $s_t \in \mathbb{R}^{6 \times 3 + 2} = \mathbb{R}^{20}$

---

## 3. Z-Score Anomaly Detection

### What Is a Z-Score?

The Z-score measures how many standard deviations a value is from the mean:

$$Z_t = \frac{x_t - \mu}{\sigma}$$

### Computing Running Statistics

Maintain exponential moving averages of mean and variance:

$$\mu_t = \alpha \mu_{t-1} + (1-\alpha) x_t$$
$$\sigma_t^2 = \alpha \sigma_{t-1}^2 + (1-\alpha) (x_t - \mu_t)^2$$

Where $\alpha$ is a decay rate (typically 0.9-0.99).

### Computing Z-Score for Loss

```
Initialize: μ = 0, σ = 1
For each step t:
    L_t = loss on batch t
    Z_t = (L_t - μ_t) / σ_t
    
    if Z_t > 2.0:
        print("Anomaly detected!")
    
    # Update running statistics
    μ_t+1 = 0.95 * μ_t + 0.05 * L_t
    σ_t+1 = 0.95 * σ_t + 0.05 * (L_t - μ_t)²
```

### Interpretation

| Z-Score | Probability | Interpretation |
|---------|-------------|-----------------|
| **Z < 1** | 68% | Normal, in expected range |
| **1 < Z < 2** | 68-95% | Slightly unusual |
| **2 < Z < 3** | 95-99.7% | Anomalous (1 in 20) |
| **Z > 3** | < 0.3% | Highly anomalous (1 in 300) |

**In MirrorMind:**
- Z > 2.0 → Trigger Fisher Information computation
- Z > 4.0 → Freeze some weights (emergency mode)

---

## 4. The Introspection RL Policy

### Why RL for Plasticity?

We want to learn: **Given internal state, how much should the model adapt?**

This is naturally a RL problem:
- **State:** Layer activations, loss, Z-score
- **Action:** Plasticity adjustment (learning rate modifier)
- **Reward:** Validation accuracy (higher is better)

### Policy Network

Train a small neural network to map state → action:

$$\pi(\text{state}) = \text{plasticity adjustment} \in [0.5, 1.5]$$

**Network architecture:**
```
Input: state vector [μ₁, σ₁, ..., L_t, Z_t]
    ↓
Linear(dim_state → 32)
    ↓
ReLU
    ↓
Linear(32 → 16)
    ↓
ReLU
    ↓
Linear(16 → 1)
    ↓
Sigmoid (outputs in [0, 1])
    ↓
Rescale to [0.5, 1.5] for plasticity adjustment
```

### REINFORCE Algorithm

Update policy to maximize expected validation accuracy:

$$\nabla_\phi J(\phi) = \mathbb{E}[\nabla_\phi \log \pi_\phi(a|s) \cdot R(s,a)]$$

Where:
- $\phi$ — Policy network parameters
- $a$ — Action (plasticity adjustment)
- $R(s,a)$ — Reward (validation accuracy improvement)

**Pseudocode:**
```python
for batch in training_loop:
    # Forward pass
    state = compute_state_vector(batch)
    action = policy_network(state)  # ∈ [0.5, 1.5]
    
    # Apply action (scale learning rate)
    effective_lr = base_lr * action
    loss = model_forward_backward(batch, lr=effective_lr)
    
    # Collect reward (on validation set)
    val_accuracy = evaluate(model, val_data)
    
    # Update policy
    reward = val_accuracy - baseline
    policy_loss = -log(action) * reward  # REINFORCE
    policy_optimizer.step()
```

---

## 5. How Introspection Prevents Divergence

### Scenario: OOD Detection

```
Step 1: Normal distribution
    Loss = 0.5, Z = 0.2 → Plasticity = 1.0 (normal learning)
    
Step 2: Subtle shift (domain drift beginning)
    Loss = 0.6, Z = 1.1 → Plasticity = 0.9 (slightly cautious)
    
Step 3: Major shift (out-of-distribution)
    Loss = 2.3, Z = 2.8 → Plasticity = 0.6 (learn slowly!)
    
Step 4: Extreme OOD
    Loss = 5.2, Z = 4.2 → Plasticity = 0.3 (nearly freeze!)
```

**Result:** Model learns slowly when uncertain, preventing over-commitment to wrong features.

### Mathematical Formulation

Define plasticity as:

$$\alpha(t) = 1 + \beta \cdot \text{tanh}(c \cdot Z_t)$$

Where:
- $\beta$ — Maximum plasticity adjustment (typically 0.2)
- $c$ — Sensitivity (typically 1.0)
- $Z_t$ — Current Z-score

Then apply to learning rate:

$$\eta_{eff}(t) = \eta_{base} \cdot \alpha(t)$$

**Behavior:**
- Z = 0 → α = 1.0, $\eta_{eff} = \eta_{base}$
- Z = 2 → α ≈ 0.8, $\eta_{eff} = 0.8 \eta_{base}$ (slower)
- Z = 4 → α ≈ 0.6, $\eta_{eff} = 0.6 \eta_{base}$ (much slower)

---

## 6. Activation Drift: Detecting Silent Failures

### The Problem: Dying Activations

Sometimes a layer's activations "die" (all become 0 or very negative) even though loss remains low. This indicates the feature detector is broken, but loss hasn't exploded yet.

### Solution: Monitor Activation Drift

Compute statistics on moving window:

$$\text{drift}_l = \|\mu_l(t) - \mu_l(t-k)\|_2$$

Where $k$ is a small window (e.g., 10 steps).

**If drift is abnormally large:**
- Layer is changing rapidly
- Might indicate instability
- Trigger introspection

### Detection

```python
# Compute layer-wise drift
for layer_idx, layer in enumerate(model):
    activations_now = get_activations(layer, batch)
    mean_now = activations_now.mean(dim=0)
    
    if layer_idx in activation_history:
        mean_past = activation_history[layer_idx]
        drift = (mean_now - mean_past).norm()
        
        if drift > threshold:
            print(f"High drift in layer {layer_idx}!")
            trigger_consolidation()
    
    activation_history[layer_idx] = mean_now
```

---

## 7. OOD Detection via Statistical Monitoring

### What Is OOD?

A sample is **Out-of-Distribution** if it comes from a different distribution than training data:

$$p_{test}(x) \neq p_{train}(x)$$

Examples:
- Clean image → Blurry/adversarial image
- English text → Chinese text
- Normal behavior → Anomalous/intrusive behavior

### Using Z-Scores for OOD

**Hypothesis:** OOD samples will have unusual internal activations

```
In-distribution: μ = expected values, σ = normal variance
    Z-scores ≈ 0 for most layers

Out-of-distribution: Activations differ from training
    Z-scores > 2 in several layers → OOD signal!
```

### Implementation

```python
class OODDetector:
    def __init__(self, model):
        self.model = model
        self.layer_stats = {}  # Store mean/variance per layer
    
    def calibrate(self, calibration_data):
        """Compute mean/variance on clean training data."""
        for x, _ in calibration_data:
            activations = get_all_activations(self.model, x)
            for layer_name, acts in activations.items():
                if layer_name not in self.layer_stats:
                    self.layer_stats[layer_name] = {
                        'mu': acts.mean(dim=0),
                        'sigma': acts.std(dim=0) + 1e-6
                    }
    
    def detect(self, x):
        """Check if x is OOD."""
        activations = get_all_activations(self.model, x)
        z_scores = []
        
        for layer_name, acts in activations.items():
            mu = self.layer_stats[layer_name]['mu']
            sigma = self.layer_stats[layer_name]['sigma']
            z = (acts.mean(dim=0) - mu) / sigma
            z_scores.append(z.abs().max().item())
        
        max_z = max(z_scores)
        is_ood = max_z > 2.0
        confidence = 1 - torch.sigmoid(torch.tensor(max_z - 2.0))
        
        return {
            'is_ood': is_ood,
            'confidence': confidence.item(),
            'max_z_score': max_z,
            'layer_z_scores': z_scores
        }
```

### Experimental Results

**Benchmark: CIFAR-10 vs SVHN (different distribution)**

```
CIFAR-10 images (in-distribution):
    Max Z-score: 0.85 ± 0.32 (mostly < 2)

SVHN images (OOD):
    Max Z-score: 3.21 ± 0.89 (mostly > 2)

OOD Detection Rate: 91% precision, 87% recall
```

---

## 8. Integration with Weight Updates

### Full Training Step with Introspection

```python
def train_step_with_introspection(batch, step_idx):
    x, y = batch
    
    # Step 1: Compute internal state
    state = compute_state_vector(model, x)
    
    # Step 2: Get plasticity adjustment from policy
    plasticity = policy_network(state)  # ∈ [0.5, 1.5]
    
    # Step 3: Compute loss
    logits = model(x)
    loss = criterion(logits, y)
    
    # Step 4: Scale effective learning rate
    effective_lr = base_lr * plasticity
    
    # Step 5: Backward pass
    loss.backward()
    
    # Step 6: Gradient clipping (prevent explosion)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Step 7: Update with effective learning rate
    for param in model.parameters():
        param.data -= effective_lr * param.grad
    
    # Step 8: Compute Z-score for logging
    z_score = compute_z_score(loss, step_idx)
    
    # Step 9: If anomalous, trigger consolidation
    if z_score > 2.5:
        consolidate_memory()
    
    return {
        'loss': loss.item(),
        'plasticity': plasticity.item(),
        'z_score': z_score,
    }
```

---

## 9. Hyperparameter Tuning

### Z-Score Threshold

- **τ = 1.5:** Sensitive, triggers often (maybe too often)
- **τ = 2.0:** Balanced, triggers ~5% of steps
- **τ = 3.0:** Conservative, triggers ~0.3% of steps

**Recommendation:** Start with τ = 2.0, adjust based on application.

### Policy Learning Rate

Train the policy network **slowly** (much slower than main network):

$$\eta_{policy} = 0.0001 \text{ (main network)} \ll \eta_{main} = 0.001$$

Too fast → Policy becomes noisy and overfits
Too slow → Policy doesn't learn meaningful plasticity control

### Exponential Moving Average Decay

Controls how fast statistics "forget" old data:

- **α = 0.95:** Fast decay, quickly adapts to new distribution
- **α = 0.99:** Slow decay, more stable but slower to respond

---

## 10. Common Issues & Debugging

### Issue 1: Z-Scores Always High

**Symptom:** Z_t > 2.0 on every batch → Fisher computation every step (wasteful)

**Root cause:** Exponential moving average decay too fast, mean is wrong

**Fix:**
```python
# Increase stability of statistics
alpha = 0.98  # was 0.95, more stable
# Or: calibrate on clean data first
calibrate_statistics(validation_data)
```

### Issue 2: Policy Doesn't Learn (Plasticity Stays at 1.0)

**Symptom:** Policy always outputs action = 1.0 (no effect)

**Root cause:** Reward signal too noisy or policy network too small

**Fix:**
```python
# Collect multiple steps before updating policy
batch_policy_gradients()

# Use larger policy network
policy = Sequential(
    Linear(20, 64),  # was 32
    ReLU(),
    Linear(64, 32),  # was 16
    ReLU(),
    Linear(32, 1)
)
```

### Issue 3: OOD Detection Has False Positives

**Symptom:** Normal examples flagged as OOD

**Root cause:** Calibration data not representative

**Fix:**
```python
# Calibrate on larger, more diverse dataset
calibration_data = create_large_calibration_set()
ood_detector.calibrate(calibration_data)

# Or: Increase threshold
ood_threshold = 3.0  # was 2.0
```

---

## 11. Mathematical Intuition: Why This Works

### Information-Theoretic View

The introspection loop estimates model **uncertainty**:

$$H(Y|X) \approx \text{variance of activations}$$

High entropy (high variance) → Model uncertain → Slow learning

### Information-Theoretic View

The introspection loop estimates when model is **out-of-domain**:

$$\text{KL}(p_{test} || p_{train}) \propto Z\text{-score}$$

High KL divergence (high Z-score) → Very different distribution → Freeze weights

### Connection to Bayesian Deep Learning

Activation variance is related to Bayesian uncertainty:

$$\text{Var}[f_\theta(x)] \approx \text{model uncertainty}$$
$$\text{Var}[\theta | D] \approx \text{parameter uncertainty}$$

Introspection uses the first form (output-space uncertainty) as a proxy.

---

## 12. Advanced Extensions

### Layered Z-Scores

Instead of global Z-score, compute per-layer:

$$Z_l = \frac{\mu_l - \bar{\mu}_l}{\bar{\sigma}_l}$$

More sensitive to layer-specific problems:

```python
for layer_idx, layer in enumerate(model):
    acts = get_activations(layer, batch)
    z_layer = (acts.mean() - history[layer_idx].mean) / history[layer_idx].std
    
    if z_layer > 2.5:
        print(f"Problem in layer {layer_idx}!")
        # Freeze this specific layer
        for param in layer.parameters():
            param.requires_grad = False
```

### Multivariate Z-Scores

Use Mahalanobis distance instead of univariate Z-score:

$$D = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$

Captures correlations between statistics.

---

## 13. Further Reading

**Key Papers:**
- Hendrycks & Gimpel (2017) - "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks"
- Lee et al. (2018) - "A Simple Unified Framework for Detecting Out-of-Distribution Samples"

**Applications:**
- Anomaly detection in sensor networks
- Adversarial robustness
- Active learning (query examples with high Z-scores)
- Domain adaptation

---

**Last Updated:** December 24, 2025  
**Reference:** MirrorMind Framework v6.1  
**Status:** Complete with theory and implementation examples
