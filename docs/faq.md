
# AirborneHRS: The Definitive Knowledge Base

> "The only documentation you will ever need."

This document covers 100% of the questions regarding the **Airborne.HRS** framework, from basic usage to theoretical physics of neural networks.

---

## 📚 Table of Contents
1.  **Philosophy & "The Why"**
2.  **Installation & Requirements**
3.  **Basic Usage (The wrapper)**
4.  **Memory Systems (EWC, SI, Replay)**
5.  **Dreaming & Sleep**
6.  **Consciousness & Introspection**
7.  **Performance & Optimization**
8.  **Compatibility (LLMs, RL, Vision)**
9.  **Troubleshooting & Debugging**
10. **Advanced Customization**

---

## 1. Philosophy & "The Why"

### Q: "Why does this package exist? Why can't I just fine-tune?"
**A:** Fine-tuning destroys the past. It's called **Catastrophic Forgetting**.
If you train a model to recognize Cats, then fine-tune it on Dogs, it becomes a Dog-Expert but completely forgets Cats.
**AirborneHRS** forces the model to retain the "Cat Neurons" while finding *new* neurons for Dogs. It turns a static model into a **Living Intelligence**.

### Q: "Is this AGI (Artificial General Intelligence)?"
**A:** It's a stepping stone.
Standard AI is "Frozen". AGI must be "Fluid". This framework provides the **Fluidity** (Continuous Learning) and **Self-Awareness** (Introspection) required for AGI-level adaptability.

---

## 2. Installation & Requirements

### Q: "What hardware do I need?"
**A:**
*   **Minimum:** CPU (Slow, but works). 8GB RAM.
*   **Recommended:** NVIDIA GPU with 8GB+ VRAM.
*   **Heavy Duty:** 24GB+ VRAM for large buffers + 4K video dreams.

### Q: "Does it work on Windows/Linux/Mac?"
**A:** Yes, it is pure Python/PyTorch.

---

## 3. Basic Usage

### Q: "How do I save/load my 'Conscious' agent?"
**A:**
*   **Save:** `torch.save(agent.state_dict(), 'brain.pt')`. This saves the Model Weights + Memory Matrices (Fisher Info) + Buffer State.
*   **Load:** `agent.load_state_dict(torch.load('brain.pt'))`.
*   **Warning:** The file size will be larger than a normal model because it carries its "Memories".

### Q: "Can I switch tasks manually?"
**A:** You calculate the boundaries implicitly, but you can force consolidation:
`agent.memory.consolidate(buffer, force=True)`.

---

## 4. Memory Systems (Deep Dive)

### Q: "What is EWC exactly?"
**A:** **Elastic Weight Consolidation**.
*   Imagine the weights are balls connected by springs (elastics) to their optimal position for Task A.
*   If a weight needs to move for Task B, the spring pulls back.
*   **Strong Spring:** Critical weight for Task A.
*   **Weak Spring:** Useless weight for Task A (free to move).
*   **Lambda (`ewc_lambda`):** The stiffness of the springs. Higher = More Memory, Less Learning.

### Q: "What is SI (Synaptic Intelligence)?"
**A:** It's EWC's faster cousin.
Instead of calculating the massive Fisher Matrix at the end, it tracks the *path integral* of weight changes during training. It knows which synapse contributed most to the drop in loss.

### Q: "Why use Hybrid (Both)?"
**A:**
*   **SI** is "Online" (Fast, tracks instantly).
*   **EWC** is "Offline" (Precise, calculated at task end).
*   **Hybrid:** Combines the speed of SI with the precision of EWC for maximum retention.

---

## 5. Dreaming & Sleep

### Q: "My model is dreaming too much! (Loss spikes)"
**A:**
*   **Symptom:** You implemented a new task, but the model keeps trying to satisfy old tasks, spiking the loss.
*   **Fix:** Increase `dream_interval` (e.g., from 10 to 100). Let it focus on reality for a bit before hallucinating.

### Q: "Can I visualize the dreams?"
**A:**
Yes! Access `agent.memory.buffer`.
It contains raw tensors. You can use `matplotlib` to plot the images stored in the hippocampus.

---

## 6. Consciousness & Introspection

### Q: "Does it feel pain?"
**A:** It feels **Mathematics**.
*   **Surprise:** High Variance in predictions. It interprets this as "Pain/Urgency" to increase learning rate.
*   **Boredom:** Low Variance. It effectively "sleeps" to save compute.

### Q: "What is the 'IntrospectionEngine' doing?"
**A:** It is a separate, smaller neural network observing the activations of the main brain. It learns to predict *how confused* the main brain is. It's a "Brain watching a Brain".

---

## 7. Performance

### Q: "My training loop is 5x slower."
**A:**
1.  **Check Buffer:** If you store 10,000 4K images, your RAM trashing is the bottleneck. Reduce `buffer_size`.
2.  **Check Dreams:** If `dream_interval=1`, you are doubling the compute (1 real step + 1 dream step). Set it to 10.
3.  **Check EWC:** Calculating the Fisher Matrix (`consolidate`) is heavy. Do it only at ends of tasks, not every epoch.

### Q: "Does it support Mixed Precision (AMP)?"
**A:** **Yes.** It respects the passing of `scaler` if you use standard PyTorch AMP loops.

### Q: "Does it work with Distributed (DDP)?"
**A:** **Experimental.**
Each GPU will have its own Buffer. This is actually good (diverse memories). But synchronization of the Fisher Matrix across GPUs is complex. Stick to Single-GPU for stability unless you are an expert.

---

## 8. Compatibility (LLMs, RL, Vision)

### Q: "Can I wrap a Transformer (LLM)?"
**A:** **Yes.**
*   Inputs are Sequences.
*   Targets are Next-Token Indices.
*   The `UniversalAdapter` handles the shapes (B, S, E).
*   **Benefit:** You can teach a customized Chatbot new facts without it forgetting its original grammar.

### Q: "Can I use it for RL (PPO/DQN)?"
**A:** **Yes.**
*   Feed `reward` into `train_step`.
*   The memory buffer acts as the Experience Replay.
*   The "Dreams" act as Off-Policy updates.

---

## 9. Troubleshooting

### Q: "Loss is NaN (Exploded)."
**A:**
*   The autonomic system usually catches this (`HealthMonitor`).
*   If it fails: Reduce `learning_rate`. EWC gradients can be huge if `lambda` is too high (e.g., >10,000).

### Q: "It forgot Task A completely."
**A:**
1.  Is `ewc_lambda` too low? (Try 5000).
2.  Is `buffer_size` too small? (Maybe old memories fell out).
3.  Did you `consolidate`? Memory is only locked AFTER consolidation. Run `agent.memory.consolidate(...)`.

---

## 10. Advanced Customization

### Q: "I want to plug in my own Memory Algorithm (e.g., GEM)."
**A:**
*   Extend `UnifiedMemoryHandler`.
*   Override `learn_from_buffer` to implement Gradient Episodic Memory (GEM) constraints instead of simple replay.

### Q: "How do I tune the 'Neuroplasticity' curve?"
**A:**
*   Edit `config.plasticity_gamma`.
*   Higher = More reactive to Surprise (Learns faster when confused).
*   Lower = More stubborn (Stable).

---

**(End of FAQ. If you have a question not listed here, you are officially a Pioneer.)**
