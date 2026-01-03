"""
Self-Awareness Demo (Framework v7.x)
======================================
This example demonstrates the functionality of the `EnhancedConsciousnessCore`
module within the `AdaptiveFramework`.

The script will:
1. Train a simple model on a single task.
2. Enable the consciousness module in the framework's configuration.
3. During training, print the "self-awareness" metrics produced by the module,
such as the current emotion, learning multiplier, and surprise score.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Import the real, functional components from the airbornehrs library
from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig

def run_consciousness_demo():
    """
    Demonstrates the live output of the consciousness module during training.
    """
    print("="*60)
    print("🚀 MirrorMind Self-Awareness (Consciousness V2) Demo")
    print("="*60)

    # --- 1. Setup: Model, Data, and Config ---

    # A simple model for the demo
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 128)
            self.fc2 = nn.Linear(128, 10)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    # Use CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleNet().to(device)
    
    # Create dummy data
    x_train = torch.randn(200, 64)
    y_train = torch.randint(0, 10, (200,))
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32)

    # Configure the framework to enable the consciousness module
    config = AdaptiveFrameworkConfig(
        learning_rate=1e-3,
        enable_consciousness=True,  # This is the key setting for this demo
        device=device
    )

    # Wrap the model in the AdaptiveFramework
    framework = AdaptiveFramework(user_model=model, config=config)
    print(f"Framework initialized on '{device}' with Consciousness enabled.\n")

    # --- 2. Training and Observation ---
    
    print("Starting training loop. Observing consciousness metrics per step...")
    print("-" * 60)
    
    step_count = 0
    for epoch in range(3):
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # The train_step method automatically invokes the consciousness module
            metrics = framework.train_step(x_batch, target_data=y_batch)
            
            # The consciousness module's outputs are available if it's enabled.
            # The `train_step` in core.py passes them through.
            # Let's get the latest metrics from the consciousness instance itself.
            if framework.consciousness and hasattr(framework.consciousness, 'last_metrics'):
                conscious_metrics = framework.consciousness.last_metrics
                
                emotion = conscious_metrics.get('emotion', 'N/A')
                multiplier = conscious_metrics.get('learning_multiplier', 1.0)
                surprise = conscious_metrics.get('surprise', 0.0)

                # Display the live "thoughts" of the model
                print(
                    f"Step {step_count:03d} | "
                    f"Emotion: {emotion:<12} | "
                    f"Multiplier: {multiplier:.2f}x | "
                    f"Surprise: {surprise:5.2f}"
                )
            
            step_count += 1
            if step_count >= 50: # Keep the demo short
                break
        if step_count >= 50:
            break

    print("-" * 60)
    print("\nDemo finished.")
    print("Observe how the 'Emotion' and 'Learning Multiplier' change based on the 'Surprise' score.")
    print(" - High surprise often leads to 'anxious' or 'curious' states with a >1.0x multiplier.")
    print(" - Low surprise can lead to a 'bored' state with a <1.0x multiplier.")
    print("This demonstrates how the framework dynamically adjusts its own learning process.")
    print("="*60)


if __name__ == "__main__":
    run_consciousness_demo()