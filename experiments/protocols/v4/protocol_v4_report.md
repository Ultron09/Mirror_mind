# Protocol V4 — Design & Mathematics

This document explains the `protocol_v4.py` implementation, its intended
uses (pathfinder CNN bot, LLM inference provider, robot controller, and
virtual employee lifecycle), and the mathematical foundations behind
its core functions.

Contents:
- Overview
- Components
- Key functions and formulas
- How to use in real systems
- Notes and next steps

---

## Overview

Protocol V4 is a practical, modular scaffold that can be used as the
starting point for deploying agents in real applications. The design is
intentionally minimal (for clarity) and is meant to be extended with
domain-specific models, datasets, and runtime systems.

This document explains the math and intended behaviour of each
component so engineers can plug in production-quality components while
retaining correct interfaces and semantics.

---

## Components

1. PerceptionCNN
   - A small convolutional model mapping local perceptual grids
     to action logits. Input shape: `[B, C, H, W]`, output: `[B, A]`.
   - Loss used during supervised training: CrossEntropyLoss for
     discrete actions. If used as a regression head, MSE is applicable.

2. A* Planner
   - A deterministic path-finding routine on a discrete grid.
   - Cost model: each step has unit cost; heuristic uses Manhattan
     distance. Produces a sequence of coordinates.

3. LLMProvider (simulator)
   - Simulates latency and cost of an LLM inference provider.
   - Useful to measure QoS (latency) and cost trade-offs for tasks
     that rely on remote inference.

4. RobotController (simulator)
   - Very small physics-free kinematic walker. Useful for testing
     control loops and integration with perception/planning.

5. VirtualEmployee
   - A soft-state agent representing an employee's skill and
     experience progression. A simple reward-driven update rule
     approximates learning and promotion.

6. EWC Penalty Function (utility)
   - Implements the Elastic Weight Consolidation penalty:

     L_ewc(\theta) = (\lambda / 2) * \sum_i F_i (\theta_i - \theta_i^*)^2

   - \theta_i: current parameter values
   - \theta_i^*: parameter anchors saved after previous tasks
   - F_i: Fisher information estimate for parameter i

---

## Key Functions and Mathematics

Below are the most important calculations and their role in the
protocol.

1) Cross-Entropy Loss (classification)

   For logits z \in R^C and target class y \in {1..C}:

   L_ce(z, y) = -log( softmax(z)_y )

   Implementation: PyTorch `torch.nn.CrossEntropyLoss()` which applies
   `log_softmax` and negative-log-likelihood.

2) Mean Squared Error (MSE) — used by the consciousness / Fisher proxy

   For predictions p and targets t of same shape:

   L_mse = (1/N) * \sum (p - t)^2

   When computing Fisher via squared gradients in the repository, a
   MSE surrogate is used for simplicity. True Fisher requires the log
   likelihood under the model; the squared-gradients proxy is a
   practical approximation.

3) Elastic Weight Consolidation (EWC)

   EWC adds a regularizer to protect parameters important to previous
   tasks. The penalty used is:

   L_ewc(\theta) = (\lambda / 2) * \sum_i F_i (\theta_i - \theta_i^*)^2

   - \theta_i are model parameters after learning the new task.
   - \theta_i^* are anchored parameters (saved copy from previous
     task).
   - F_i is the estimated Fisher information for parameter i.

   The protocol implements Fisher by accumulating squared gradients
   across a small replay sample S from the feedback buffer:

   F_i \approx (1/|S|) \sum_{x \in S} (\partial/\partial \theta_i L(x; \theta^*))^2

   Once F is computed and \theta^* saved, the penalty is computed in
   each subsequent training step and added to the loss.

4) Planner cost model

   For A* on a discrete grid, path cost is simply the sum of step
   costs (we use 1 per move). Heuristic h uses Manhattan: h(a,b) =
   |a_x - b_x| + |a_y - b_y|.

   A* guarantees an optimal solution for admissible heuristics.

5) Virtual employee learning dynamics

   The simple update rule in the demo is:

   skill_{t+1} = skill_t + \alpha * reward
   experience_{t+1} = experience_t + |reward|

   Where \alpha is a small learning rate (0.1 by default). Levels
   (intern/junior/mid/senior) are thresholded on accumulated
   experience. This is not a realistic human learning model but serves
   to show a lifecycle transition driven by performance signals.

---

## How to use Protocol V4 in real systems

1. Replace the simple `PerceptionCNN` with a production-grade
   perception model (e.g. ResNet, MobileNet) and integrate dataset
   loaders for real sensor inputs.

2. Swap the `LLMProvider` simulator with a production inference
   client (e.g. call to OpenAI, LlamaCPP, or a local model server).
   Capture actual latency, token usage, and error rates.

3. Replace the `RobotController` with a proper kinematic/dynamic
   controller connected to the robot's control stack or simulator
   (e.g. Gazebo, PyBullet, or real hardware APIs).

4. For continual learning across tasks, collect replay examples in a
   buffer and compute Fisher information as implemented in the
   repository EWC handler. Tune \lambda (EWC strength) to avoid
   under- or over-constraining weights.

5. For workforce automation (virtual employee), treat the VirtualEmployee
   class as a scaffolding: replace the simple reward with domain
   driven KPIs, use richer skill representations, and integrate HR
   lifecycle tooling (onboarding, reviews, promotion rules).

---

## Example: End-to-end usage pattern (Pathfinder + Robot)

1. Perception CNN produces action logits from a local observation.
2. Planner computes a coarse global path (A*) given environment map.
3. Controller tracks along the path, querying perception for local
   corrections and replanning on obstacle detection.
4. After task completion, collect traces and compute Fisher from the
   buffer to consolidate knowledge with EWC.

Mathematically, the per-step training objective for perception could be:

L_total = L_task + L_ewc
where L_task = CrossEntropy( logits, action )
and L_ewc computed as above.

---

## Notes and next steps

- The demo scripts in `protocol_v4.py` output synthetic reports in
  `protocol_v4_report.json` and per-mode JSON files. Replace the
  synthetic data with real inputs to get practical metrics.
- Tune EWC hyperparameters (\lambda) and Fisher sampling size.
- Integrate the protocol with `airbornehrs` MirrorMind components if
  you want the advanced adapter/meta-controller features.

---

If you'd like, I can now:

- Hook `protocol_v4` into the MirrorMind `MirrorMindSystem` (so it uses
  the project's EWC and meta-controller out-of-the-box).
- Create a CI test that runs the protocol on a small synthetic task
  and verifies outputs.
- Expand the VirtualEmployee model to use a parameterized skill
  network (so promotions depend on learned features rather than
  simple thresholds).

Tell me which follow-up you prefer and I'll implement it next.
