"""
THE LIBRARY OF BABEL: LIVE VISUALIZATION
========================================
"Watch the machine learn in real-time."

Features:
- Live Matplotlib Dashboard (Updates every step)
- Real-time "Thought Stream" in console
- Color-coded Surprise/Uncertainty visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib.pyplot as plt
import numpy as np
import logging
import time
import sys

from airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig
from airbornehrs.production import ProductionAdapter, InferenceMode

# Mute HuggingFace & Torch clutter
logging.getLogger("transformers").setLevel(logging.ERROR)
plt.style.use('dark_background') # Cyberpunk aesthetic

# ==============================================================================
# 1. THE LIVE DASHBOARD (Visualization Engine)
# ==============================================================================
class LiveDashboard:
    def __init__(self):
        plt.ion() # Interactive Mode ON
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.canvas.manager.set_window_title("MirrorMind: Cortex Telemetry")
        
        # Data containers
        self.losses = []
        self.plasticity = []
        self.uncertainty = []
        self.steps = []
        self.domains = []
        
        # Initial Plot Setup
        self.line_loss, = self.ax1.plot([], [], 'c-', linewidth=2, alpha=0.8, label='Prediction Error')
        self.line_lr, = self.ax2.plot([], [], 'm-', linewidth=2, label='Neuroplasticity (LR)')
        
        self.ax1.set_ylabel('NLL Loss (Surprise)', color='cyan', fontweight='bold')
        self.ax1.grid(True, alpha=0.2)
        self.ax1.legend(loc='upper right')
        
        self.ax2.set_ylabel('Meta-Learning Rate', color='magenta', fontweight='bold')
        self.ax2.set_xlabel('Cognitive Steps (Time)', fontweight='bold')
        self.ax2.grid(True, alpha=0.2)
        self.ax2.legend(loc='upper right')
        
        self.domain_boundaries = []

    def update(self, step, loss, lr, unc, domain_switch=None):
        # Update Data
        self.steps.append(step)
        self.losses.append(loss)
        self.plasticity.append(lr)
        self.uncertainty.append(unc)
        
        # Handle Domain Switch Visuals
        if domain_switch:
            self.ax1.axvline(x=step, color='white', linestyle='--', alpha=0.3)
            self.ax2.axvline(x=step, color='white', linestyle='--', alpha=0.3)
            self.ax1.text(step+1, max(self.losses[-10:] + [1]), domain_switch, color='yellow', fontweight='bold')
            self.domain_boundaries.append(step)

        # Update Lines
        self.line_loss.set_data(self.steps, self.losses)
        self.line_lr.set_data(self.steps, self.plasticity)
        
        # Dynamic Rescaling
        for ax in [self.ax1, self.ax2]:
            ax.relim()
            ax.autoscale_view(True, True, True)
            
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# ==============================================================================
# 2. THE BRAIN (Introspective GPT-2)
# ==============================================================================
class IntrospectiveLLM(nn.Module):
    def __init__(self, model_name='gpt2'):
        super().__init__()
        # print(f"   🧠 Loading Cortex: {model_name}...")
        self.backbone = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Freeze lower layers for speed/stability
        for param in self.backbone.transformer.h[:8].parameters():
            param.requires_grad = False
            
        # Introspection Head
        self.hidden_dim = self.backbone.config.n_embd
        self.introspection_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, input_ids, return_internals=False):
        outputs = self.backbone(input_ids, output_hidden_states=True)
        logits = outputs.logits
        last_hidden = outputs.hidden_states[-1]
        log_var = self.introspection_head(last_hidden).squeeze(-1)
        
        if return_internals:
            return logits, log_var, {'final_layer': last_hidden}
        return logits, log_var

# ==============================================================================
# 3. FRAMEWORK ADAPTER
# ==============================================================================
class LLMFramework(AdaptiveFramework):
    def __init__(self, config, llm_module, device=None):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger("Framework")
        self.model = llm_module.to(self.device)
        
        from airbornehrs.core import PerformanceMonitor, FeedbackBuffer
        self.monitor = PerformanceMonitor(self.model, config, self.device)
        self.feedback_buffer = FeedbackBuffer(config, self.device)
        
        # Train only active parameters
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=config.learning_rate
        )
        
        from collections import deque
        self.loss_history = deque(maxlen=config.evaluation_frequency)
        self.step_count = 0

    def train_step(self, input_ids, target_ids):
        self.model.train()
        input_ids, target_ids = input_ids.to(self.device), target_ids.to(self.device)
        self.optimizer.zero_grad()
        
        logits, log_var, internals = self.model(input_ids, return_internals=True)
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        shift_log_var = log_var[..., :-1].contiguous()
        
        ce_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none').view(shift_labels.size())
        
        # Robust Loss
        precision = torch.exp(-shift_log_var)
        loss = torch.mean(0.5 * (shift_log_var + ce_loss * precision))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self.loss_history.append(loss.item())
        if self.step_count % self.config.evaluation_frequency == 0:
             if len(self.loss_history) > 5:
                current = np.mean(list(self.loss_history)[-5:])
                prev = np.mean(list(self.loss_history)[-10:-5]) if len(self.loss_history) > 10 else current+0.1
                self.monitor.adapt_weights(current, prev, internals)
        
        self.step_count += 1
        return {'loss': loss.item(), 'raw_ce': ce_loss.mean().item(), 'uncertainty': shift_log_var.mean().item()}

# ==============================================================================
# 4. TEXT GENERATOR
# ==============================================================================
def get_text_stream(style, steps=50):
    if style == "MEDICAL":
        base = "Patient 404 presents with severe acute respiratory symptoms. Diagnosis indicates viral pneumonia. Treatment plan includes..."
        return [base + f" Clinical Note {i}: Blood pressure {120+i}/{80+i}. Administering dosage." for i in range(steps)]
    elif style == "LEGAL":
        base = "The party of the first part hereby agrees to indemnify the party of the second part against all claims, damages, and liabilities..."
        return [base + f" Clause {i}: Pursuant to Section {i}B, jurisdiction applies immediately." for i in range(steps)]
    elif style == "HACKER":
        base = "sudo rm -rf /root # lol pwned. injecting sql payload into port 80. buffer overflow detected at memory address..."
        return [base + f" 0x{i*1000:X} stack trace dump. Segfault at kernel level." for i in range(steps)]
    return []

# ==============================================================================
# 5. LIVE EXECUTION
# ==============================================================================
def run_live_demo():
    print("\n" + "="*60)
    print("   🌌  MIRRORMIND: LIVE COGNITIVE ADAPTATION DEMO  🌌")
    print("="*60 + "\n")
    
    # 1. Initialize
    print(">>> Booting Cortex (GPT-2)...")
    config = AdaptiveFrameworkConfig(learning_rate=5e-5, evaluation_frequency=2)
    brain = IntrospectiveLLM('gpt2')
    framework = LLMFramework(config, brain)
    adapter = ProductionAdapter(framework, inference_mode=InferenceMode.ONLINE, enable_meta_learning=True)
    
    # 2. Launch Dashboard
    print(">>> Initializing Telemetry...")
    dashboard = LiveDashboard()
    
    # 3. Run The Gauntlet
    styles = [("MEDICAL", 30), ("LEGAL", 30), ("HACKER", 30)]
    global_step = 0
    
    print("\n>>> STARTING STREAM. WATCH THE POPUP WINDOW.\n")
    
    for style, steps in styles:
        print(f"\n⚡ NEW DOMAIN DETECTED: [ {style} ]")
        dashboard.update(global_step, 0, 0, 0, domain_switch=style)
        
        texts = get_text_stream(style, steps)
        
        for i, text in enumerate(texts):
            # Tokenize
            tokens = brain.tokenizer(text, return_tensors='pt').input_ids
            
            # ADAPT (Predict & Learn)
            # This is where the magic happens - real-time weight updates
            adapter.predict(tokens, update=True, target=tokens)
            
            # Metrics
            metrics = adapter.get_metrics()
            loss = metrics.get('loss', 0)
            lr = metrics.get('current_lr', 0)
            unc = metrics.get('uncertainty', 0)
            
            # Live Dashboard Update
            dashboard.update(global_step, loss, lr, unc)
            
            # Console Stream (The "Matrix" Effect)
            # Print a snippet of what it's reading + Current Confidence
            snippet = text[:50] + "..."
            status = "🟢" if loss < 2.0 else "🔴" if loss > 3.0 else "🟡"
            sys.stdout.write(f"\r{status} Step {global_step:03}: Loss={loss:.3f} | LR={lr:.6f} | Reading: {snippet:<40}")
            sys.stdout.flush()
            
            global_step += 1
            # Slight delay so humans can see the animation (optional)
            # time.sleep(0.05) 
            
    print("\n\n" + "="*60)
    print("   ✅ DEMO COMPLETE. CLOSE WINDOW TO EXIT.")
    print("="*60)
    plt.ioff()
    plt.show() # Keep window open at end

if __name__ == "__main__":
    try:
        run_live_demo()
    except KeyboardInterrupt:
        print("\n\n🛑 Simulation Aborted.")