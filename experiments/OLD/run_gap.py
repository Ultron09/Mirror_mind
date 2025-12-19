"""
THE EVOLUTION GAP V2: Curriculum-Based Capability Acquisition (Standalone)
==========================================================================
A self-contained experiment to bridge the reasoning gap between GPT-2 and GPT-3.

Modules:
1. Data Generator (Curriculum Logic)
2. Dashboard (Real-time Visualization)
3. Subject (Introspective GPT-2)
4. Framework (Adaptive Trainer)
5. Execution (The Experiment Loop)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib.pyplot as plt
import numpy as np
import random
import logging
import sys
import re

# MirrorMind Core Imports
from airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig
from airbornehrs.production import ProductionAdapter, InferenceMode

# Clean UI
logging.getLogger("transformers").setLevel(logging.ERROR)
plt.style.use('dark_background')

# ==============================================================================
# MODULE 1: DATA & CURRICULUM
# ==============================================================================
def generate_curriculum_batch(difficulty):
    """
    Generates arithmetic problems based on difficulty level.
    Level 1: Single digit (0-9)
    Level 2: Double digit (10-50)
    """
    if difficulty == 1:
        a, b = random.randint(0, 9), random.randint(0, 9)
    else:
        a, b = random.randint(10, 50), random.randint(10, 50)
        
    ans = str(a + b)
    # Few-shot prompt to enforce format
    prompt = f"Math: 1+1=2\nMath: 2+2=4\nMath: {a}+{b}= A: {ans}" 
    return [prompt], [ans]

def check_accuracy_robust(decoded_text, target_ans):
    """
    Robustly checks answer using Regex (ignores whitespace/newlines).
    Looks for pattern 'A: <number>'
    """
    match = re.search(r"A:\s*(\d+)", decoded_text)
    if match:
        prediction = match.group(1)
        return 1.0 if prediction == target_ans else 0.0
    return 0.0

# ==============================================================================
# MODULE 2: VISUALIZATION DASHBOARD
# ==============================================================================
class GapDashboardV2:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self.fig.canvas.manager.set_window_title("MirrorMind: Evolution Gap V2")
        
        self.steps = []
        self.accuracies = []
        self.levels = []
        
        # Plot Lines
        self.line_acc, = self.ax.plot([], [], 'r-', linewidth=3, label='Evolved Accuracy')
        self.line_lvl, = self.ax.plot([], [], 'c--', alpha=0.5, label='Task Difficulty (Scaled)')
        
        # Benchmarks
        self.ax.axhline(y=80.0, color='gold', linestyle='--', label='GPT-3 Target')
        self.ax.axhline(y=10.0, color='gray', linestyle=':', label='Baseline')
        
        self.ax.set_ylim(0, 100)
        self.ax.set_ylabel('Accuracy (%)', fontweight='bold', color='white')
        self.ax.set_xlabel('Evolution Steps', fontweight='bold', color='white')
        self.ax.legend(loc='upper left')
        self.ax.grid(True, alpha=0.2)

    def update(self, step, acc, level):
        self.steps.append(step)
        self.accuracies.append(acc)
        self.levels.append(level * 20) # Scale level 1->20, 2->40 for visibility
        
        self.line_acc.set_data(self.steps, self.accuracies)
        self.line_lvl.set_data(self.steps, self.levels)
        
        self.ax.set_xlim(0, max(len(self.steps), 10))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# ==============================================================================
# MODULE 3: THE SUBJECT (Model Definition)
# ==============================================================================
class EvolvingBrain(nn.Module):
    """
    A GPT-2 model augmented with an Introspection Head.
    Defined locally to ensure modularity.
    """
    def __init__(self):
        super().__init__()
        self.backbone = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Introspection Head (Estimates Uncertainty)
        self.monitor = nn.Sequential(
            nn.Linear(768, 64),
            nn.GELU(),
            nn.Linear(64, 1) # Log Variance
        )
        
        # Unfreeze specific layers for plasticity
        # We unfreeze the last MLP block and the monitor
        for name, param in self.backbone.named_parameters():
            if "mlp" in name and "11" in name: 
                param.requires_grad = True
            elif "monitor" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, input_ids, return_internals=False):
        out = self.backbone(input_ids, output_hidden_states=True)
        hidden = out.hidden_states[-1] 
        
        # Calculate uncertainty
        uncertainty = self.monitor(hidden).squeeze(-1)
        
        if return_internals:
            return out.logits, uncertainty, {'h': hidden}
        return out.logits, uncertainty

# ==============================================================================
# MODULE 4: TRAINING FRAMEWORK
# ==============================================================================
class MathFramework(AdaptiveFramework):
    """
    Custom training logic for arithmetic reasoning.
    Overrides the default train_step to handle Tokenizer logic.
    """
    def __init__(self, config, model):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.logger = logging.getLogger("MathFrame")
        
        # Initialize internal monitors
        from airbornehrs.core import PerformanceMonitor
        self.monitor = PerformanceMonitor(self.model, config, self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=config.learning_rate
        )
        self.step_count = 0
        self.loss_history = []

    def train_step(self, input_ids, target_ids):
        self.model.train()
        input_ids = input_ids.to(self.device)
        
        self.optimizer.zero_grad()
        logits, log_var, internals = self.model(input_ids, return_internals=True)
        
        # Standard Next-Token Prediction Loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        loss = F.cross_entropy(shift_logits.view(-1, 50257), shift_labels.view(-1))
        
        # Backward Pass
        loss.backward()
        self.optimizer.step()
        
        # Trigger Adaptation (Meta-Controller hook)
        self.loss_history.append(loss.item())
        if self.step_count % 3 == 0 and len(self.loss_history) > 5:
             current = np.mean(self.loss_history[-3:])
             prev = np.mean(self.loss_history[-6:-3])
             # This adjusts weights based on introspective signals
             self.monitor.adapt_weights(current, prev, internals)
             
        self.step_count += 1
        return {'loss': loss.item()}

# ==============================================================================
# MODULE 5: EXECUTION LOOP
# ==============================================================================
def run_gap_v2():
    print("\n🧬 INITIALIZING EVOLUTION GAP V2 (Curriculum-Assisted)")
    print("   Goal: Bridge the genetic gap between GPT-2 and GPT-3 via Test-Time Training.")
    
    # 1. Configuration
    config = AdaptiveFrameworkConfig(
        learning_rate=2e-4, 
        weight_adaptation_lr=1e-3,
        evaluation_frequency=1
    )
    
    # 2. Initialization
    brain = EvolvingBrain()
    framework = MathFramework(config, brain)
    adapter = ProductionAdapter(framework, inference_mode=InferenceMode.ONLINE, enable_meta_learning=True)
    dashboard = GapDashboardV2()
    
    history_acc = []
    
    # 3. Evolution Loop
    for step in range(200):
        # Curriculum: Switch to Level 2 (Harder) at step 100
        level = 1 if step < 100 else 2
        
        # A. Generate Data
        prompts, answers = generate_curriculum_batch(level)
        tokens = brain.tokenizer(prompts[0], return_tensors='pt').input_ids
        
        # B. Measure Zero-Shot Accuracy (Before Update)
        with torch.no_grad():
            # Generate answer tokens
            out = brain.backbone.generate(tokens[:, :-3], max_new_tokens=5, pad_token_id=50256)
            decoded = brain.tokenizer.decode(out[0], skip_special_tokens=True)
            # Check correctness
            acc = check_accuracy_robust(decoded, answers[0])
            
        # C. Adapt (The Learning Step)
        adapter.predict(tokens, update=True, target=tokens)
        
        # D. Visualization & Logging
        history_acc.append(acc * 100.0)
        smooth_acc = np.mean(history_acc[-10:]) if len(history_acc) > 10 else np.mean(history_acc)
        
        dashboard.update(step, smooth_acc, level)
        
        # Status Bar
        bar = "█" * int(smooth_acc/5)
        print(f"\rStep {step:03} [Lvl {level}] | Acc: {smooth_acc:5.1f}% | Target: {answers[0]} | {bar}", end="")

    print("\n\n✅ V2 Experiment Complete.")
    print("   Results saved to 'evolution_gap_v2.png'")
    plt.savefig("evolution_gap_v2.png")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_gap_v2()