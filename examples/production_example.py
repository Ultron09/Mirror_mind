"""
Production Integration Example
==============================

Demonstrates how to integrate airbornehrs into a real application
where the model learns continuously from incoming data.
"""

import torch
import torch.nn as nn
from airbornehrs import (
    AdaptiveFramework,
    AdaptiveFrameworkConfig,
    MetaController,
    ProductionAdapter,
    InferenceMode
)


# ==================== STEP 1: TRAINING ====================

def train_phase():
    """Phase 1: Initial training with meta-learning"""
    print("=" * 60)
    print("PHASE 1: TRAINING WITH ADAPTIVE META-LEARNING")
    print("=" * 60)
    
    # Create configuration
    config = AdaptiveFrameworkConfig(
        model_dim=128,
        num_layers=4,
        num_heads=4,
        batch_size=32,
        learning_rate=1e-3,
        epochs=5
    )
    
    # Initialize framework
    framework = AdaptiveFramework(config)
    
    print(f"\n✅ Framework initialized")
    print(f"  Device: {framework.device}")
    print(f"  Model parameters: {sum(p.numel() for p in framework.model.parameters()):,}")
    
    # Generate synthetic training data
    # In production: load from data source
    num_samples = 100
    seq_len = 10
    X_train = torch.randn(num_samples, seq_len, config.model_dim)
    y_train = torch.randn(num_samples, seq_len, config.model_dim)
    
    # Training loop
    print(f"\n🚀 Training for {config.epochs} epochs...")
    for epoch in range(config.epochs):
        framework.model.train()
        
        total_loss = 0
        for i in range(0, len(X_train), config.batch_size):
            batch_x = X_train[i:i+config.batch_size]
            batch_y = y_train[i:i+config.batch_size]
            
            metrics = framework.train_step(batch_x, batch_y)
            total_loss += metrics['loss']
        
        avg_loss = total_loss / (len(X_train) // config.batch_size)
        print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    # Save checkpoint
    checkpoint_path = "production_model.pt"
    framework.save_checkpoint(checkpoint_path)
    print(f"\n✅ Model saved to {checkpoint_path}")
    
    return checkpoint_path


# ==================== STEP 2: PRODUCTION DEPLOYMENT ====================

def production_phase(checkpoint_path: str):
    """Phase 2: Deployment with online learning"""
    print("\n" + "=" * 60)
    print("PHASE 2: PRODUCTION DEPLOYMENT WITH ONLINE LEARNING")
    print("=" * 60)
    
    # Load model in production mode
    print(f"\n📦 Loading model from {checkpoint_path}...")
    adapter = ProductionAdapter.load_checkpoint(
        checkpoint_path,
        inference_mode=InferenceMode.ONLINE  # Enable online learning
    )
    print(f"✅ Model loaded in {InferenceMode.ONLINE} mode")
    
    # Simulate incoming data stream
    print(f"\n🌊 Processing data stream (online learning enabled)...")
    
    for step in range(10):
        # Simulate new data arriving
        new_batch = torch.randn(1, 10, 128)
        new_target = torch.randn(1, 10, 128)
        
        # Make prediction and optionally learn
        prediction = adapter.predict(
            new_batch,
            update=True,  # Enable online learning
            target=new_target
        )
        
        # Get uncertainty estimate
        uncertainty = adapter.get_uncertainty(new_batch)
        
        if step % 3 == 0:
            metrics = adapter.get_metrics()
            print(f"  Step {step:2d}: Loss = {metrics.get('avg_recent_loss', 0):.4f}, "
                  f"Uncertainty = {uncertainty.mean():.4f}")
    
    print(f"\n✅ Online learning complete")
    
    # Save updated model
    adapter.save_checkpoint("production_model_updated.pt")
    print(f"✅ Updated model saved")


# ==================== STEP 3: ADVANCED USAGE ====================

def advanced_example():
    """Advanced: Using MetaController for custom adaptation"""
    print("\n" + "=" * 60)
    print("PHASE 3: ADVANCED ADAPTATION WITH META-CONTROLLER")
    print("=" * 60)
    
    # Initialize framework
    config = AdaptiveFrameworkConfig(
        model_dim=128,
        num_layers=4,
        num_heads=4
    )
    framework = AdaptiveFramework(config)
    
    # Attach meta-controller for advanced adaptation
    meta_controller = MetaController(framework)
    
    print(f"\n✅ MetaController attached")
    
    # Training with explicit adaptation
    print(f"\n🎯 Training with explicit meta-learning...")
    
    X_train = torch.randn(50, 10, 128)
    y_train = torch.randn(50, 10, 128)
    
    for epoch in range(3):
        framework.model.train()
        
        # Training step
        metrics = framework.train_step(X_train, y_train)
        
        # Meta-learning adaptation
        performance_metrics = {
            'loss_improvement': 0.01 if epoch > 0 else 0.0
        }
        
        adaptation_metrics = meta_controller.adapt(
            loss=metrics['loss'],
            performance_metrics=performance_metrics
        )
        
        print(f"  Epoch {epoch}: Loss = {metrics['loss']:.4f}, "
              f"LR = {adaptation_metrics['learning_rate']:.2e}, "
              f"Difficulty = {adaptation_metrics.get('curriculum_difficulty', 0):.2f}")
    
    # Get adaptation summary
    summary = meta_controller.get_summary()
    print(f"\n✅ Adaptation summary:")
    print(f"  Steps: {summary['step_count']}")
    print(f"  Current LR: {summary['current_lr']:.2e}")
    print(f"  Curriculum difficulty: {summary['curriculum_difficulty']:.2f}")


# ==================== MAIN ====================

if __name__ == "__main__":
    print("\n" + "🎯" * 30)
    print("AIRBORNEHRS PRODUCTION INTEGRATION EXAMPLE")
    print("🎯" * 30)
    
    # Phase 1: Training
    checkpoint_path = train_phase()
    
    # Phase 2: Production with online learning
    production_phase(checkpoint_path)
    
    # Phase 3: Advanced usage
    advanced_example()
    
    print("\n" + "✅" * 30)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("✅" * 30)
    print("\nNext steps:")
    print("1. Replace synthetic data with your real data")
    print("2. Adjust config parameters (model_dim, num_layers, etc.)")
    print("3. Choose appropriate InferenceMode (STATIC, ONLINE, BUFFERED)")
    print("4. Deploy to production!")
    print()
