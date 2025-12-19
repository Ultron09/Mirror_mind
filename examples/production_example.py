"""
MirrorMind Quick Start Guide
============================
A simple "Train -> Serve" example for AirborneHRS.
"""

import torch
from airbornehrs import (
    AdaptiveFramework,
    AdaptiveFrameworkConfig,
    ProductionAdapter,
    InferenceMode
)

def main():
    print("🚀 MirrorMind Quick Start Initiated...")

    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    config = AdaptiveFrameworkConfig(
        model_dim=64,           # Size of the model
        num_layers=2,           # Depth of the brain
        learning_rate=0.001,    # Base speed of learning
        batch_size=16,
        epochs=3
    )

    # ==========================================
    # 2. THE LAB (Training Phase)
    # ==========================================
    print("\n[PHASE 1] Training Model...")
    
    # Initialize the Framework (The Brain)
    framework = AdaptiveFramework(config)
    
    # Create Dummy Data (Input -> Target)
    X_train = torch.randn(100, 10, 64) 
    y_train = torch.randn(100, 10, 64)

    # Train Loop
    for epoch in range(config.epochs):
        # framework.train_step() handles forward pass, loss, AND introspection
        metrics = framework.train_step(X_train, y_train)
        print(f"   Epoch {epoch+1}: Loss = {metrics['loss']:.4f}")

    # Save the "Brain"
    framework.save_checkpoint("my_model.pt")
    print("   ✅ Model saved to 'my_model.pt'")


    # ==========================================
    # 3. THE WILD (Production Phase)
    # ==========================================
    print("\n[PHASE 2] Deploying to Production...")

    # Load into Production Adapter
    # InferenceMode.ONLINE enables continuous learning from new data
    adapter = ProductionAdapter.load_checkpoint(
        "my_model.pt",
        inference_mode=InferenceMode.ONLINE 
    )
    print("   ✅ Adapter loaded. Online Learning: ENABLED")

    # Simulate Live Data Stream
    new_data = torch.randn(1, 10, 64)   # User Input
    ground_truth = torch.randn(1, 10, 64) # Feedback (or Self-Correction)

    print("\n   Incoming Request...")
    
    # Run Prediction + Learn (One Step)
    # update=True triggers the Meta-Controller to adjust weights instantly
    output = adapter.predict(new_data, update=True, target=ground_truth)

    # Check Vitals
    metrics = adapter.get_metrics()
    print(f"   📊 Prediction Complete.")
    print(f"      Current Uncertainty: {metrics.get('uncertainty_mean', 0.0):.4f}")
    print(f"      Plasticity Rate:     {metrics.get('current_lr', 0.0):.6f}")

if __name__ == "__main__":
    main()