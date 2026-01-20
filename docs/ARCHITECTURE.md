
# AirborneHRS Architecture: Memory & Retention

This document explains the technical mechanisms behind the framework's "Conscious Memory" capabilities.

## 1. How Task Retention Works (The Anti-Forgetting Shield)
The system uses a hybrid of **Elastic Weight Consolidation (EWC)** and **Synaptic Intelligence (SI)**.

*   **Concept:** Think of the neural network weights as a ball rolling down a hill (finding the lowest error). When we learn Task A, the ball settles in a valley. When we move to Task B, the ball wants to roll to a new valley, often leaving the first one (forgetting).
*   **The Solution:** We attach an "Elastic Tether" to the weights that were critical for Task A.
    *   If a weight was important for A, it becomes "heavy" and hard to change.
    *   If a weight was unimportant for A (it can change without hurting A), it remains "light" and free to adapt for Task B.
*   **Implementation:**
    *   **Synaptic Importance:** We calculate the gradient magnitude (Fisher Information) during training.
    *   **Penalty Loss:** `Loss = Task_B_Loss + Lambda * (Current_Weight - Old_Weight)^2 * Importance`

## 2. Where is "Memory" Stored?

### A. Short-Term Memory (Hippocampus)
*   **Implementation:** `FeedbackBuffer` / `PrioritizedExperienceReplay`.
*   **Location:** **RAM (System Memory)** during execution.
*   **Content:** It stores a sliding window of recent input samples (Images, Audio, Text) and their Targets.
*   **Capacity:** Configurable (e.g., last 10,000 samples). Old memories are overwritten by new ones unless marked as "High Surprise" (Emotional tagging).

### B. Long-Term Memory (Cortex)
*   **Implementation:** `UnifiedMemoryHandler`.
*   **Location:** **Model Weights & Checkpoint Files (`.pt`)**.
*   **Content:** The "Memory" is not a database of images. It is encoded **implicitly** in the neural connections (Synapses).
    *   Explicit storage: The `Fisher Information Matrix` (tensors representing weight importance) is saved alongside the model weights in the checkpoint file.
    *   When you load a checkpoint, you restore these "Synaptic Constraints".

## 3. What are "Dreams"?
*   **Definition:** Dreaming is an active process (`learn_from_buffer`), not a stored file.
*   **Mechanism:**
    1.  Every `dream_interval` steps (e.g., 50), the model pauses training on real data.
    2.  It samples a mixed batch of memories from the `FeedbackBuffer`.
    3.  It performs a training step on these memories.
*   **Purpose:** This strengthens the neural pathways for past events, "consolidating" them from Short-Term (Buffer) to Long-Term (Weights) memory, preventing them from fading.
