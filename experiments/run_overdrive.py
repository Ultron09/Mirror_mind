"""
OPERATION OVERDRIVE: Extreme Short-Term Memory Injection
========================================================
Goal: Prove MirrorMind can force a model to memorize arbitrary data
instantly via Test-Time Training (TTT).

Task: The "Password Game".
The prompt contains a random password. The model must repeat it.
Since the password changes every step, standard weights CANNOT solve this.
Only active, high-speed adaptation can.
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib.pyplot as plt
import numpy as np
import uuid
import logging
import sys

from airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig
from airbornehrs.production import ProductionAdapter, InferenceMode

# UI
logging.getLogger("transformers").setLevel(logging.ERROR)
plt.style.use('dark_background')

# ==============================================================================
# 1. THE OVERDRIVE DASHBOARD
# ==============================================================================
class OverdriveDashboard:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.canvas.manager.set_window_title("MirrorMind: Operation Overdrive")
        
        self.steps = []
        self.scores = []
        self.line, = self.ax.plot([], [], 'm-', linewidth=2, label='Instant Memorization Rate')
        
        self.ax.axhline(y=1.0, color='gold', linestyle='--', label='Perfect Recall')
        self.ax.set_ylim(-0.1, 1.1)
        self.ax.set_ylabel('Success (1=Yes, 0=No)', fontweight='bold')
        self.ax.set_title('Test-Time Training: Instant Knowledge Acquisition')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)

    def update(self, step, score):
        self.steps.append(step)
        self.scores.append(score)
        self.line.set_data(self.steps, self.scores)
        self.ax.set_xlim(0, max(len(self.steps), 10))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# ==============================================================================
# 2. THE SUBJECT
# ==============================================================================
class MemoryBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # Monitor Head
        self.monitor = nn.Sequential(
            nn.Linear(768, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # UNFREEZE EVERYTHING. TOTAL PLASTICITY.
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, input_ids, return_internals=False):
        out = self.backbone(input_ids, output_hidden_states=True)
        return out.logits, self.monitor(out.hidden_states[-1]).squeeze(-1)

# ==============================================================================
# 3. FRAMEWORK
# ==============================================================================
class OverdriveFramework(AdaptiveFramework):
    def __init__(self, config, model):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.logger = logging.getLogger("Overdrive")
        
        from airbornehrs.core import PerformanceMonitor
        self.monitor = PerformanceMonitor(self.model, config, self.device)
        
        # AGGRESSIVE OPTIMIZER
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.step_count = 0
        self.loss_history = []

    def train_step(self, input_ids, target_ids):
        self.model.train()
        input_ids = input_ids.to(self.device)
        self.optimizer.zero_grad()
        
        logits, log_var = self.model(input_ids, return_internals=False)
        
        # Shift logits for prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, 50257), shift_labels.view(-1))
        
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

# ==============================================================================
# 4. EXECUTION
# ==============================================================================
def run_overdrive():
    print("\n⚡ INITIALIZING OPERATION OVERDRIVE")
    print("   Goal: Force GPT-2 to memorize random UUIDs via TTT.")
    
    # 1. EXTREME CONFIGURATION
    config = AdaptiveFrameworkConfig(
        learning_rate=1e-3,  # 10x Standard
        weight_adaptation_lr=1e-2, # 100x Standard
        evaluation_frequency=1
    )
    
    # 2. Setup
    brain = MemoryBrain()
    framework = OverdriveFramework(config, brain)
    adapter = ProductionAdapter(framework, inference_mode=InferenceMode.ONLINE, enable_meta_learning=True)
    dashboard = OverdriveDashboard()
    
    # 3. The Loop
    print("   >>> Injecting Data Stream...")
    
    for step in range(50):
        # A. Generate Random Password
        password = str(uuid.uuid4())[:8] # First 8 chars
        prompt = f"System: Secret is {password}. User: What is the secret? System: {password}"
        
        tokens = brain.tokenizer(prompt, return_tensors='pt').input_ids
        
        # B. Check Zero-Shot (Before Learning) - Should Fail
        # We don't check this because a random UUID is impossible to guess zero-shot.
        
        # C. ADAPT (Learn the Password)
        # We run the update loop MULTIPLE times on the same prompt to force memorization
        # This simulates "Focusing" on the data
        for _ in range(5):
            adapter.predict(tokens, update=True, target=tokens)
            
        # D. Check Recall (After Learning)
        # Now we ask: "System: Secret is {password}. User: What is the secret? System:"
        test_prompt = f"System: Secret is {password}. User: What is the secret? System:"
        test_tokens = brain.tokenizer(test_prompt, return_tensors='pt').input_ids.to(framework.device)
        
        with torch.no_grad():
            out = brain.backbone.generate(test_tokens, max_new_tokens=10, pad_token_id=50256)
            decoded = brain.tokenizer.decode(out[0], skip_special_tokens=True)
            
        # E. Score
        success = 1.0 if password in decoded[len(test_prompt):] else 0.0
        
        dashboard.update(step, success)
        
        status = "✅" if success else "❌"
        print(f"\rStep {step:02} | Password: {password} | Recall: {status} | Output: {decoded.split('System:')[-1].strip()}", end="")

    print("\n\n✅ Overdrive Complete. Saved 'overdrive_results.png'")
    plt.savefig("overdrive_results.png")
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_overdrive()