"""
Production Simulation: The "Extreme Life-Long Learning" Test
============================================================
Goal: Prove that MirrorMind improves OVER TIME in production, 
even after a very short (1 epoch) initial training phase.

Scenario:
1. "Lazy Training": Train for just 1 Epoch (Underfitting).
2. "The Gauntlet": Deploy to Production with shifting domains (NLP->CV->Audio).
3. "Disturbance": Add random noise/corruption to inputs.
4. "Proof": Measure Instant Loss vs. Cumulative Error.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
from dataclasses import dataclass

from airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig
from airbornehrs.meta_controller import MetaControllerConfig
from airbornehrs.production import ProductionAdapter, InferenceMode

# ==============================================================================
# 1. UNIVERSAL SENSORS (The Interface)
# ==============================================================================
class UniversalProjector(nn.Module):
    """Maps Text/Image/Audio to the Brain's Embedding Space"""
    def __init__(self, model_dim: int):
        super().__init__()
        self.text_embed = nn.Embedding(1000, model_dim) # NLP
        self.vision_proj = nn.Conv2d(1, model_dim, 4, 4) # CV (28x28 -> 7x7)
        self.audio_proj = nn.Linear(64, model_dim)       # Audio
        
    def forward(self, x, domain):
        if domain == 'NLP': return self.text_embed(x.long())
        if domain == 'CV':  return self.vision_proj(x).flatten(2).transpose(1, 2)
        if domain == 'AUDIO': return self.audio_proj(x)
        return x

# ==============================================================================
# 2. DATA GENERATOR (The World)
# ==============================================================================
def get_stream_batch(domain, step, batch_size=1, disturbance=False):
    """Generates streaming data with optional 'Extreme Disturbance'"""
    
    if domain == 'NLP':
        # Pattern: Predict next token (Simple arithmetic sequence)
        # Disturbance: Random token insertions
        seq = torch.arange(step, step+10).unsqueeze(0).repeat(batch_size, 1) % 1000
        if disturbance and random.random() < 0.3:
            seq = torch.randint(0, 1000, seq.shape) # Chaos!
        return seq, seq # Target = Input (Auto-regression)

    elif domain == 'CV':
        # Pattern: Moving bar
        img = torch.zeros(batch_size, 1, 28, 28)
        col = step % 28
        img[:, :, :, col] = 1.0 
        # Disturbance: Pixel Dropout
        if disturbance:
            mask = torch.rand_like(img) > 0.5
            img = img * mask
        return img, img

    elif domain == 'AUDIO':
        # Pattern: Sine wave frequency shift
        freq = (step % 50) + 1
        x = torch.linspace(0, 10, 50).unsqueeze(0).repeat(batch_size, 1)
        wave = torch.sin(freq * x).unsqueeze(2).repeat(1, 1, 64)
        # Disturbance: White Noise
        if disturbance:
            wave += torch.randn_like(wave) * 0.5
        return wave, wave

# ==============================================================================
# 3. PHASE 1: "LAZY" TRAINING
# ==============================================================================
def run_lazy_training():
    print("\n🏭 PHASE 1: LAZY TRAINING (1 Epoch Only)")
    print("   Testing if Stabilizer allows rapid convergence...")
    
    config = AdaptiveFrameworkConfig(
        model_dim=64, num_layers=2, learning_rate=0.005,
        compile_model=False # Disable for small test script
    )
    
    framework = AdaptiveFramework(config, device='cpu')
    projector = UniversalProjector(64)
    
    # Train on just 50 steps of NLP (Very little!)
    losses = []
    for i in range(50):
        x, y = get_stream_batch('NLP', i, batch_size=16)
        x_emb, y_emb = projector(x, 'NLP'), projector(y, 'NLP')
        
        metrics = framework.train_step(x_emb, y_emb)
        losses.append(metrics['loss'])
    
    print(f"   ✅ Initial Loss: {losses[0]:.4f} -> Final Loss: {losses[-1]:.4f}")
    
    # Save checkpoint
    framework.save_checkpoint("lazy_model.pt")
    torch.save(projector.state_dict(), "projector.pt")
    return "lazy_model.pt"

# ==============================================================================
# 4. PHASE 2: PRODUCTION (The Gauntlet)
# ==============================================================================
def run_production_simulation(checkpoint_path):
    print("\n🌍 PHASE 2: PRODUCTION DEPLOYMENT (Online Learning)")
    print("   Scenario: NLP -> CV -> Audio (With Disturbance)")
    
    # Load Brain
    adapter = ProductionAdapter.load_checkpoint(
        checkpoint_path, 
        inference_mode=InferenceMode.ONLINE, # <--- ACTIVE LEARNING
        device='cpu'
    )
    
    # Load Eyes/Ears
    projector = UniversalProjector(64)
    projector.load_state_dict(torch.load("projector.pt"))
    
    history = {'loss': [], 'plasticity': []}
    
    # Define The Gauntlet
    scenarios = [
        ('NLP', 100, False),   # Normal NLP
        ('CV', 100, True),     # Visual data + NOISE (Disturbance)
        ('AUDIO', 100, False)  # Audio data
    ]
    
    global_step = 0
    
    for domain, steps, disturbance in scenarios:
        print(f"\n   ⚡ INCOMING STREAM: {domain} (Disturbance={disturbance})")
        
        for i in range(steps):
            # 1. Get Live Data
            x, y = get_stream_batch(domain, i, disturbance=disturbance)
            x_emb = projector(x, domain)
            y_emb = projector(y, domain)
            
            # 2. Predict & Learn (Online)
            # The adapter handles the Meta-Controller logic internally!
            adapter.predict(x_emb, update=True, target=y_emb)
            
            # 3. Measure "After-Learning" Performance (Did it improve?)
            # We check the metrics *after* the update to see stability
            metrics = adapter.get_metrics()
            
            loss = metrics.get('avg_recent_loss', 0)
            lr = metrics.get('current_lr', 0)
            
            history['loss'].append(loss)
            history['plasticity'].append(lr)
            
            if i % 25 == 0:
                print(f"     Step {i}: Loss={loss:.4f} | Plasticity={lr:.5f}")
                
    return history

# ==============================================================================
# 5. VERIFICATION PLOTS
# ==============================================================================
def verify_results(history):
    loss = np.array(history['loss'])
    lr = np.array(history['plasticity'])
    
    plt.figure(figsize=(10, 8))
    
    # Plot 1: Does it improve? (Loss Trend)
    plt.subplot(2, 1, 1)
    plt.plot(loss, label='Production Error', color='blue', alpha=0.6)
    # Trend line
    z = np.polyfit(range(len(loss)), loss, 1)
    p = np.poly1d(z)
    plt.plot(loss, p(range(len(loss))), "r--", label='Improvement Trend')
    
    plt.title("Proof of Online Improvement (Decreasing Error Trend)")
    plt.ylabel("Prediction Error")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Is the Stabilizer Working? (LR Spikes)
    plt.subplot(2, 1, 2)
    plt.plot(lr, label='Neuroplasticity (LR)', color='green')
    plt.title("Stabilizer Reflex (Response to Domain Shifts)")
    plt.xlabel("Production Inference Steps")
    plt.ylabel("Learning Rate")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("production_proof.png")
    print("\n✅ Verification complete. Saved 'production_proof.png'")
    
    # Final Verdict
    if loss[-1] < loss[0]:
        print("\n🏆 RESULT: SUCCESS. Model improved over time!")
    else:
        print("\n⚠️ RESULT: WARNING. Model did not improve overall.")

if __name__ == "__main__":
    ckpt = run_lazy_training()
    hist = run_production_simulation(ckpt)
    verify_results(hist)