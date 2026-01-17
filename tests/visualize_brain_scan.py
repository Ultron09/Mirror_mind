
import sys
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Force local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig

class VisualCortex(nn.Module):
    def __init__(self):
        super().__init__()
        # 3x3 Convolution = 9 weights per neuron
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 100) # Prefrontal

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def run_brain_scan():
    torch.manual_seed(42)
    device = 'cpu'
    
    # Configure Brain
    cfg = AdaptiveFrameworkConfig(device=device)
    cfg.memory_type = 'hybrid'
    cfg.ewc_lambda = 2000.0
    
    model = AdaptiveFramework(VisualCortex(), cfg, device=device)
    
    # 1. CREATE MEMORIES (Train on Task A)
    print(">>> Imprinting Memories (Training)...")
    inputs = torch.randn(32, 1, 32, 32)
    targets = torch.randint(0, 10, (32,))
    
    for _ in range(10):
        model.train_step(inputs, target_data=targets)
        
    model.memory.consolidate(feedback_buffer=model.prioritized_buffer)
    print(">>> Memories Consolidated (Fisher Matrix Computed).")

    # 3. EXTRACT "CONNECTOME"
    fisher_map = None
    keys = list(model.memory.fisher_dict.keys())
    print(f"DEBUG: Available Fisher Keys: {keys}")
    
    target_key = 'model.fc.weight'
    if target_key not in keys:
        # Search for any weight (2D)
        for k in keys:
            if 'weight' in k and 'fc' in k:
                target_key = k
                break
    
    if target_key in keys:
        fisher_map = model.memory.fisher_dict[target_key].detach().cpu().numpy()
        print(f"DEBUG: Selected '{target_key}' with shape {fisher_map.shape}")
        
        # Ensure 2D
        if len(fisher_map.shape) == 1:
            # Reshape or unsqueeze
            d = int(np.sqrt(fisher_map.shape[0]))
            if d*d == fisher_map.shape[0]:
                fisher_map = fisher_map.reshape(d, d)
            else:
                fisher_map = fisher_map.reshape(1, -1)
    else:
        print("DEBUG: No suitable 2D weight found for visualization.")


    # 3. EXTRACT "NEURAL FIRING" (Activations)
    # Pass a new input and hook the layer
    print(">>> Scanning Neural Activity...")
    activations = []
    def hook_fn(module, input, output):
        activations.append(output.detach().cpu().numpy())
    
    handle = model.model.fc.register_forward_hook(hook_fn)
    
    stimulus = torch.randn(1, 1, 32, 32) # A single thought
    model.model.eval()
    model(stimulus)
    handle.remove()
    
    firing_pattern = activations[0][0] # 100 neurons

    # --- PLOTTING ---
    print(">>> Rendering Brain Scan...")
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 10))
    
    # A. THE CONNECTOME (Memory Structure)
    # Reshape Flat FC weights (100x2048) to something 2D visuals can handle or just show a chunk
    ax1 = plt.subplot(2, 1, 1)
    
    if fisher_map is not None:
        # Visualize subset of synaptic strengths (Fisher)
        # Shows "Hardened" memories
        subset_fisher = fisher_map[:50, :100] # First 50 neurons, 100 synapses each
        im1 = ax1.imshow(subset_fisher, aspect='auto', cmap='magma', interpolation='nearest')
        ax1.set_title("THE HIPPOCAMPUS (Long-Term Memory Structure)", fontsize=14, color='white', pad=10)
        ax1.set_ylabel("Neuron ID")
        ax1.set_xlabel("Synaptic Connections")
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label("Synaptic Rigidity (Fisher Info)")
    else:
        ax1.text(0.5, 0.5, "No Memories Formed (Fisher Empty)", ha='center', color='white')
    
    # B. NEURAL ACTIVITY (The "Thought")
    ax2 = plt.subplot(2, 1, 2)
    # Visualize 1D firing pattern as a bar chart (Spike Train)
    ax2.bar(range(len(firing_pattern)), firing_pattern, color='#00ffbd', alpha=0.8)
    ax2.set_title("PREFRONTAL CORTEX (Real-Time Neural Firing)", fontsize=14, color='white', pad=10)
    ax2.set_xlabel("Neuron ID")
    ax2.set_ylabel("Activation Amplitude (Hz)")
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.1)
    
    # Annotate "Active" vs "Silent"
    active_count = (firing_pattern > 0).sum()
    ax2.text(0.02, 0.9, f"Active Neurons: {active_count}/100", transform=ax2.transAxes, color='white', fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), 'brain_scan.png')
    plt.savefig(save_path, dpi=150)
    print(f"Brain Scan saved to: {save_path}")

if __name__ == "__main__":
    run_brain_scan()
