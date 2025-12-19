"""
MirrorMind Quick Start Guide (Sync V2.0)
========================================
A simple "Train -> Serve" example for AirborneHRS.
Demonstrates Reptile Meta-Learning integration.
"""

import torch
import torch.nn as nn
import os
from airbornehrs import (
    AdaptiveFramework,
    AdaptiveFrameworkConfig,
    ProductionAdapter,
    InferenceMode
)

# Define a simple model to wrap
class SimpleBrain(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

def main():
    print("🚀 MirrorMind Quick Start Initiated...")

    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    BATCH_SIZE = 16
    EPOCHS = 3
    
    # Enable compilation if on Linux/CUDA, else disable for safety
    is_compile = torch.cuda.is_available() and os.name != 'nt'
    
    config = AdaptiveFrameworkConfig(
        model_dim=64,           
        num_layers=2,           
        learning_rate=0.001,
        compile_model=is_compile,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # ==========================================
    # 2. THE LAB (Training Phase)
    # ==========================================
    print(f"\n[PHASE 1] Training Model on {config.device}...")
    
    base_model = SimpleBrain(input_dim=64, output_dim=64)
    
    # Wrap it with MirrorMind (The "Consciousness" Layer)
    # NOTE: This initializes MetaController internally (Sync fix applied)
    framework = AdaptiveFramework(base_model, config)
    
    # Create Dummy Data
    X_train = torch.randn(100, 10, 64).to(config.device)
    y_train = torch.randn(100, 10, 64).to(config.device)

    # Train Loop
    for epoch in range(EPOCHS):
        total_loss = 0
        for i in range(0, len(X_train), BATCH_SIZE):
            batch_X = X_train[i:i+BATCH_SIZE]
            batch_y = y_train[i:i+BATCH_SIZE]
            
            # train_step handles:
            # 1. Forward Pass
            # 2. Introspection (Uncertainty)
            # 3. EWC Penalty (if active)
            # 4. Reptile Meta-Update
            metrics = framework.train_step(batch_X, batch_y)
            total_loss += metrics['loss']
            
        avg_loss = total_loss / (len(X_train) // BATCH_SIZE)
        print(f"   Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    # Save the "Brain"
    framework.save_checkpoint("my_model.pt")
    print("   ✅ Model saved to 'my_model.pt'")


    # ==========================================
    # 3. THE WILD (Production Phase)
    # ==========================================
    print("\n[PHASE 2] Deploying to Production...")

    # Load into Production Adapter
    # InferenceMode.ONLINE enables continuous learning (Reptile)
    adapter = ProductionAdapter.load_checkpoint(
        "my_model.pt",
        inference_mode=InferenceMode.ONLINE 
    )
    print("   ✅ Adapter loaded. Online Learning: ENABLED")

    # Simulate Live Data Stream
    new_data = torch.randn(1, 10, 64).to(config.device)
    ground_truth = torch.randn(1, 10, 64).to(config.device)

    print("\n   Incoming Request...")
    
    # Run Prediction + Learn (One Step)
    # update=True triggers the Meta-Controller to adjust weights instantly
    output = adapter.predict(new_data, update=True, target=ground_truth)

    # Check Vitals
    metrics = adapter.get_metrics()
    print(f"   📊 Prediction Complete.")
    print(f"      Current Uncertainty: {metrics.get('uncertainty_mean', 0.0):.4f}")
    
    # Robustly fetch learning rate (depends on MetaController state)
    lr = metrics.get('learning_rate', metrics.get('current_lr', 0.0))
    print(f"      Plasticity Rate:     {lr:.6f}")
    
    # Verify Reptile status
    reptile_active = metrics.get('reptile_active', False)
    print(f"      Reptile Active:      {reptile_active}")

if __name__ == "__main__":
    try:

        main()
    except Exception as e:
        print(f"❌ An error occurred: {e}")