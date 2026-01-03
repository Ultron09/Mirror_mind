"""
MirrorMind Production Example (Framework v7.x)
==============================================
A simple "Train -> Serve" example for the AirborneHRS framework.
This demonstrates saving a trained adaptive model and loading it for
online learning in a production-like setting.
"""

import torch
import torch.nn as nn
import os
from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig
from airbornehrs.production import ProductionAdapter, InferenceMode

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
    print("🚀 MirrorMind Production Example Initiated...")

    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    BATCH_SIZE = 16
    EPOCHS = 3
    MODEL_PATH = "my_adaptive_model.pt"
    
    # Use CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = AdaptiveFrameworkConfig(
        learning_rate=0.001,
        enable_consciousness=True, # Enable for metrics and dynamic learning
        memory_type='hybrid',      # Use memory protection
        device=device
    )

    # ==========================================
    # 2. THE LAB (Training Phase)
    # ==========================================
    print(f"\n[PHASE 1] Training Model on {config.device}...")
    
    base_model = SimpleBrain(input_dim=64, output_dim=10)
    
    # Wrap the model with the AdaptiveFramework
    framework = AdaptiveFramework(base_model, config)
    
    # Create Dummy Data
    x_train = torch.randn(100, 64).to(config.device)
    y_train = torch.randint(0, 10, (100,)).to(config.device)

    # Standard Training Loop
    for epoch in range(EPOCHS):
        total_loss = 0
        for i in range(0, len(x_train), BATCH_SIZE):
            batch_x = x_train[i:i+BATCH_SIZE]
            batch_y = y_train[i:i+BATCH_SIZE]
            
            # framework.train_step handles all the complexity internally
            metrics = framework.train_step(batch_x, target_data=batch_y)
            total_loss += metrics['loss']
            
        avg_loss = total_loss / (len(x_train) // BATCH_SIZE)
        print(f"   Epoch {epoch+1}/{EPOCHS}: Avg Loss = {avg_loss:.4f}")

    # Save the trained adaptive model
    framework.save_checkpoint(MODEL_PATH)
    print(f"   ✅ Model and configuration saved to '{MODEL_PATH}'")


    # ==========================================
    # 3. THE WILD (Production/Inference Phase)
    # ==========================================
    print("\n[PHASE 2] Deploying to Production...")

    # For loading, we need a fresh instance of the base model architecture
    production_model_architecture = SimpleBrain(input_dim=64, output_dim=10)

    # Load the trained framework into the ProductionAdapter
    # This correctly restores the model weights, optimizer state, and CONFIGURATION.
    adapter = ProductionAdapter.load_checkpoint(
        MODEL_PATH,
        model=production_model_architecture,
        inference_mode=InferenceMode.ONLINE # Enable continuous learning
    )
    print("   ✅ Adapter loaded. Online Learning is ENABLED.")

    # Simulate a new piece of data arriving in a live stream
    new_data = torch.randn(1, 64).to(config.device)
    ground_truth = torch.randint(0, 10, (1,)).to(config.device)

    print("\n   Receiving new data point...")
    
    # Run prediction AND learn from it in a single step
    # update=True triggers the online learning mechanism
    output = adapter.predict(new_data, update=True, target=ground_truth)

    # Check the model's vitals after the update
    metrics = adapter.get_metrics()
    print(f"   📊 Prediction and update complete.")
    print(f"      Model Emotion:     {metrics.get('emotion', 'N/A')}")
    if 'learning_multiplier' in metrics:
        print(f"      Learning Multiplier: {metrics.get('learning_multiplier'):.2f}")
    if 'learning_rate' in metrics:
        print(f"      Meta LR:             {metrics.get('learning_rate')}")
    if 'reptile_active' in metrics:
        print(f"      Reptile Active:      {metrics.get('reptile_active', False)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
