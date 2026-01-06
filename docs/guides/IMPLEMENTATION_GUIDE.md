# AirborneHRS Implementation Guide: Multi-Modal Strategies

This guide provides specific strategies for applying **AirborneHRS V8.0 "Sentient"** to various data modalities. The `AdaptiveFramework` is universal, but each modality benefits from specific configurations.

---

## 👁️ Computer Vision (CNNs, ViTs)

### Recommended Configuration
Vision models often benefit from **high plasticity** in early layers (feature extraction) and **strong memory** in later layers (classification).

```python
config = AdaptiveFrameworkConfig(
    model_dim=512,          # Match your CNN's feature dimension
    use_amp=True,           # Critical for image processing speed
    memory_type='hybrid',   # Best for retaining visual concepts
    use_moe=True,           # [V7.1] Excellent for multi-task vision
    num_experts=4,
    input_dim=3*224*224     # Required if using MoE on raw pixels (rare)
)
```

### Wrapping Strategy
Wrap the entire backbone + classifier.
```python
# Standard ResNet
backbone = torchvision.models.resnet18(pretrained=True)
backbone.fc = nn.Linear(512, num_classes)

# Wrap
agent = AdaptiveFramework(backbone, config=config)
```

### Tips
-   **Data Augmentation**: Apply standard augmentations (RandomCrop, Flip) *before* passing data to the agent. The agent learns robustly from augmented views.
-   **Introspection**: Monitor `surprise` metrics. High surprise on specific classes indicates a need for more training samples of that class.

---

## 🗣️ NLP (Transformers, LLMs)

### Recommended Configuration
NLP requires **long-term context** and **recursive reasoning** (System 2).

```python
config = AdaptiveFrameworkConfig(
    model_dim=768,          # Match Transformer hidden size (e.g., BERT base)
    num_heads=12,           # Match Transformer heads
    enable_consciousness=True, # Critical for language reasoning
    use_lookahead=True,     # Helps escape local minima in complex loss landscapes
    use_gradient_centralization=True
)
```

### Wrapping Strategy
For Hugging Face Transformers, wrap the model but ensure the `forward` signature matches.
```python
class HFWrapper(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.hf_model(input_ids, attention_mask=attention_mask)
        return outputs.logits

# Wrap
agent = AdaptiveFramework(HFWrapper(model), config=config)
```

### Tips
-   **Tokenization**: Handle tokenization outside the agent. Pass tensor indices `[Batch, SeqLen]` to `train_step`.
-   **System 2**: For complex queries, the "Thought Process" loop in V8.0 will automatically recurse, effectively performing "internal Chain-of-Thought".

---

## 🔊 Audio (Spectrograms, Waveforms)

### Recommended Configuration
Audio data is noisy. **Active Shielding** and **Noise Robustness** are key.

```python
config = AdaptiveFrameworkConfig(
    enable_active_shield=True, # Filter out background noise/anomalies
    active_shield_threshold=0.1,
    memory_type='ewc',      # EWC is often sufficient and faster for 1D signals
    consolidation_surprise_threshold=3.0 # Only memorize distinct sounds
)
```

### Wrapping Strategy
Works best with Spectrogram inputs (2D) treated like images, or 1D CNNs for raw waveforms.

### Tips
-   **Silence Removal**: Pre-process to remove silence. The agent treats silence as "low surprise" and may ignore it, but it wastes compute.
-   **OOD Detection**: The `surprise` metric is an excellent anomaly detector for audio (e.g., detecting machinery failure sounds).

---

## 🧬 Embeddings & Retrieval

### Recommended Configuration
For metric learning (Contrastive Loss, Triplet Loss), focus on **manifold stability**.

```python
config = AdaptiveFrameworkConfig(
    learning_rate=1e-4,     # Lower LR for stable embedding spaces
    meta_learning_rate=1e-5,
    use_prioritized_replay=True # Replay hard negatives
)
```

### Wrapping Strategy
Wrap the projector network.
```python
encoder = nn.Sequential(
    nn.Linear(input_dim, 256),
    nn.ReLU(),
    nn.Linear(256, embedding_dim) # Output normalized embeddings
)
agent = AdaptiveFramework(encoder, config=config)
```

### Tips
-   **Normalization**: Ensure your model outputs normalized vectors if using Cosine Similarity.
-   **Memory**: The `HolographicAssociativeMemory` is naturally suited for clustering embeddings. Use `agent.save_memory()` to persist the learned manifold.
