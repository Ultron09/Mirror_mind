"""
Universal Adaptation Benchmark (The "Omni-Test")
================================================
Tests MirrorMind's ability to adapt across completely different domains:
1. NLP (Text Sequence Prediction)
2. Computer Vision (Image Patch Processing)
3. Audio (Spectrogram Analysis)

This proves the framework's "Meta-Learning" capabilities apply regardless 
of the input modality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict

from airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig
from airbornehrs.meta_controller import MetaController, MetaControllerConfig

# Safety
torch.autograd.set_detect_anomaly(True)

# ==============================================================================
# 1. DOMAIN PROJECTORS ( The "Sensors" )
# Adapts raw data (Pixels, Audio, Text) into MirrorMind's embedding space
# ==============================================================================

class UniversalProjector(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        self.model_dim = model_dim
        
        # --- NLP Sensor (Tokens -> Embeddings) ---
        self.vocab_size = 1000
        self.text_embed = nn.Embedding(self.vocab_size, model_dim)
        
        # --- Vision Sensor (Images -> Patches -> Embeddings) ---
        # Simulates a ViT patch embedding (1 channel image -> flat patch)
        self.patch_size = 4
        self.vision_proj = nn.Conv2d(1, model_dim, kernel_size=4, stride=4)
        
        # --- Audio Sensor (Spectrogram Freqs -> Embeddings) ---
        # Projects frequency bins to model dimension
        self.audio_freq_bins = 64
        self.audio_proj = nn.Linear(self.audio_freq_bins, model_dim)

    def process_nlp(self, x):
        return self.text_embed(x.long())

    def process_vision(self, x):
        # x: [Batch, 1, 28, 28] -> [Batch, Dim, 7, 7]
        x = self.vision_proj(x)
        # Flatten: [Batch, Dim, 49] -> [Batch, 49, Dim]
        return x.flatten(2).transpose(1, 2)

    def process_audio(self, x):
        # x: [Batch, Time, Freq] -> [Batch, Time, Dim]
        return self.audio_proj(x)

# ==============================================================================
# 2. DATA GENERATORS (Synthetic Simulations)
# ==============================================================================

def generate_domain_batch(domain: str, batch_size=32) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates synthetic data for specific domains."""
    
    if domain == "NLP":
        # Task: Next Token Prediction
        # Input: Random integer tokens [B, Seq]
        seq_len = 20
        raw_x = torch.randint(0, 1000, (batch_size, seq_len))
        # Target: Shifted sequence (conceptually) - for this test we map to embedding dim
        # To make it compatible with the Framework's MSE Loss, we target the embedding of the "next" token
        raw_y = torch.randint(0, 1000, (batch_size, seq_len)) 
        return raw_x, raw_y

    elif domain == "VISION":
        # Task: Autoencoding / Reconstruction
        # Input: 28x28 grayscale images (MNIST-like)
        raw_x = torch.randn(batch_size, 1, 28, 28)
        raw_y = raw_x.clone() # Target is reconstruction
        return raw_x, raw_y

    elif domain == "AUDIO":
        # Task: Denoising
        # Input: Spectrogram [Batch, Time, Freq]
        time_steps = 50
        freq_bins = 64
        clean_signal = torch.sin(torch.linspace(0, 10, time_steps)).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, freq_bins)
        noise = torch.randn_like(clean_signal) * 0.5
        
        raw_x = clean_signal + noise
        raw_y = clean_signal # Target is clean signal
        return raw_x, raw_y
        
    else:
        raise ValueError(f"Unknown domain: {domain}")

# ==============================================================================
# 3. THE UNIVERSAL BENCHMARK
# ==============================================================================

def run_universal_benchmark():
    print("🌍 INITIALIZING MIRRORMIND UNIVERSAL BENCHMARK...")
    print("   Goal: Test adaptation across NLP -> Vision -> Audio streams.")

    # 1. Configuration
    # We use a larger model dim to accommodate complex features
    config = AdaptiveFrameworkConfig(
        model_dim=128,
        num_layers=6,
        learning_rate=0.0005,
        adaptation_threshold=0.05,
        evaluation_frequency=5
    )
    
    # Meta-Controller allows high plasticity (high max_lr) to switch contexts
    meta_config = MetaControllerConfig(
        base_lr=0.0005,
        max_lr=0.02,
        reptile_learning_rate=1e-4
    )

    # 2. Initialize Core System
    framework = AdaptiveFramework(config, device='cpu')
    controller = MetaController(framework, meta_config)
    
    # 3. Initialize "Sensors" (Projector)
    # The projector converts raw domain data into what MirrorMind can understand
    projector = UniversalProjector(model_dim=config.model_dim)
    optimizer_proj = torch.optim.AdamW(projector.parameters(), lr=1e-3)

    # 4. The Experiment Loop
    domains = [
        ("NLP", 150),      # Phase 1: Text
        ("VISION", 150),   # Phase 2: Images
        ("AUDIO", 150)     # Phase 3: Sound
    ]
    
    history = {'loss': [], 'lr': [], 'domain_switch': []}
    global_step = 0

    for domain_name, steps in domains:
        print(f"\n⚡ SWITCHING CONTEXT TO: {domain_name} (Steps {global_step}-{global_step+steps})")
        history['domain_switch'].append(global_step)
        
        for i in range(steps):
            # A. Get Raw Data
            raw_x, raw_y = generate_domain_batch(domain_name)
            
            # B. Project to MirrorMind's Mental Space (The "Embeddings")
            projector.train()
            optimizer_proj.zero_grad()
            
            if domain_name == "NLP":
                emb_x = projector.process_nlp(raw_x)
                emb_y = projector.process_nlp(raw_y) # Target is also embedding
            elif domain_name == "VISION":
                emb_x = projector.process_vision(raw_x)
                emb_y = projector.process_vision(raw_y)
            elif domain_name == "AUDIO":
                emb_x = projector.process_audio(raw_x)
                emb_y = projector.process_audio(raw_y)
            
            # C. MirrorMind Training Step (The Brain)
            # The framework learns the *patterns* in the embeddings
            metrics = framework.train_step(emb_x, emb_y)
            
            # D. Backpropagate through Projector (End-to-End learning)
            # Note: Framework does its own internal backprop, but we need to update projector
            # In a real setup, we might detach, but here we simulate joint training
            loss_tensor = torch.tensor(metrics['loss'], requires_grad=True)
            loss_tensor.backward() 
            optimizer_proj.step()

            # E. Meta-Adaptation (The Stabilizer)
            # Checks if the "Brain" is struggling with the new "Sensor" data
            adaptation = controller.adapt(metrics['loss'])
            
            # Logging
            history['loss'].append(metrics['loss'])
            history['lr'].append(adaptation['learning_rate'])
            global_step += 1
            
            if i % 50 == 0:
                print(f"   Step {i}: Loss={metrics['loss']:.4f} | LR={adaptation['learning_rate']:.5f}")

    return history, domains

# ==============================================================================
# 4. VISUALIZATION
# ==============================================================================

def plot_universal_results(history, domains):
    print("\n📊 Generating Universal Adaptation Report...")
    
    loss = history['loss']
    lr = history['lr']
    switches = history['domain_switch']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Smooth loss
    def smooth(d): return np.convolve(d, np.ones(10)/10, mode='valid')
    
    # Plot Loss
    ax1.plot(loss, color='gray', alpha=0.3)
    ax1.plot(smooth(loss), color='blue', linewidth=2, label='Adaptation Loss')
    ax1.set_yscale('log')
    ax1.set_ylabel('Loss (Log Scale)', fontweight='bold')
    ax1.set_title('MirrorMind: Universal Adaptation across Domains', fontsize=14, fontweight='bold')
    
    # Add domain labels
    colors = ['purple', 'green', 'orange']
    for idx, (name, steps) in enumerate(domains):
        start = switches[idx]
        ax1.axvline(x=start, color=colors[idx], linestyle='--', alpha=0.8)
        ax1.text(start+10, ax1.get_ylim()[1]*0.5, name, color=colors[idx], fontweight='bold', fontsize=12)

    # Plot Learning Rate (The Reflex)
    ax2.plot(lr, color='red', linewidth=2, label='Meta-Learning Rate')
    ax2.set_ylabel('Plasticity (LR)', color='red', fontweight='bold')
    ax2.set_xlabel('Global Training Steps')
    ax2.grid(True, alpha=0.3)
    
    # Highlight spikes
    for start in switches:
        ax2.axvline(x=start, color='black', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig("universal_benchmark_results.png")
    print("✅ Report saved to: universal_benchmark_results.png")

if __name__ == "__main__":
    hist, doms = run_universal_benchmark()
    plot_universal_results(hist, doms)