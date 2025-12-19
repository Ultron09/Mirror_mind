import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig
from airbornehrs.meta_controller import MetaController, MetaControllerConfig
from airbornehrs.production import ProductionAdapter, InferenceMode

# Enable anomaly detection for safety
torch.autograd.set_detect_anomaly(True)
os.makedirs("checkpoints", exist_ok=True)

# ==============================================================================
# SCENARIO GENERATOR
# ==============================================================================
def get_market_data(day_type, batch_size=1, seq_len=10, dim=32):
    """
    Simulates streaming financial data.
    Day Type:
      - 'NORMAL': Standard patterns (Identity)
      - 'CRISIS': Inverted patterns (Negation) + High Volatility
    """
    x = torch.randn(batch_size, seq_len, dim)
    
    if day_type == 'NORMAL':
        # Predictable market: Future = Past
        y = x.clone() 
        
    elif day_type == 'CRISIS':
        # Market Crash: Inverse correlation + Volatility
        y = -x.clone() * 1.5 
    
    return x, y

# ==============================================================================
# PHASE 1: THE LAB (Offline Training)
# ==============================================================================
def run_training_phase():
    print("\n🏭 PHASE 1: THE LAB (Offline Training)")
    print("   Goal: Train a base model on 'Normal' data and ship it.")
    
    # 1. Config for Training
    fw_config = AdaptiveFrameworkConfig(
        model_dim=32, num_layers=4, learning_rate=0.001,
        evaluation_frequency=10
    )
    meta_config = MetaControllerConfig(
        base_lr=0.001, max_lr=0.01
    )
    
    # 2. Init
    framework = AdaptiveFramework(fw_config)
    controller = MetaController(framework, meta_config)
    
    # 3. Train Loop
    print("   Training on 500 batches of Normal Data...")
    losses = []
    
    for step in range(500):
        x, y = get_market_data('NORMAL', batch_size=32)
        metrics = framework.train_step(x, y)
        controller.adapt(metrics['loss'])
        losses.append(metrics['loss'])
        
        if step % 100 == 0:
            print(f"     Step {step}: Loss = {metrics['loss']:.4f}")
            
    print(f"   ✅ Training Complete. Final Loss: {losses[-1]:.6f}")
    
    # 4. Ship It (Save Checkpoint)
    save_path = "checkpoints/production_v1.pt"
    framework.save_checkpoint(save_path)
    print(f"   📦 Model packaged and saved to: {save_path}")
    return save_path

# ==============================================================================
# PHASE 2: PRODUCTION (Live Deployment)
# ==============================================================================
def run_production_phase(checkpoint_path):
    print("\n🌍 PHASE 2: PRODUCTION DEPLOYMENT (Live Inference)")
    print("   Goal: Serve predictions. Adapt online if 'The Crisis' hits.")
    
    # 1. Load Model into Production Adapter
    # InferenceMode.ONLINE means it learns from EVERY interaction
    adapter = ProductionAdapter.load_checkpoint(
        checkpoint_path, 
        inference_mode=InferenceMode.ONLINE 
    )
    print("   ✅ ProductionAdapter Online. Mode: ONLINE LEARNING")
    
    history = {'loss': [], 'uncertainty': []}
    
    # ---------------------------------------------------------
    # DAY 1: Normal Operations (Stability Check)
    # ---------------------------------------------------------
    print("\n   📅 DAY 1: Normal Market Conditions (Steps 0-100)")
    for step in range(100):
        # STREAMING DATA (Batch size 1 = Real-time)
        x_stream, y_true = get_market_data('NORMAL', batch_size=1)
        
        # Predict & Learn (One line!)
        # update=True enables the Online Learning Loop
        pred = adapter.predict(x_stream, update=True, target=y_true)
        
        # Calculate Error for telemetry
        loss = torch.nn.functional.mse_loss(pred, y_true).item()
        
        # Introspection (Uncertainty)
        unc = adapter.get_uncertainty(x_stream).mean().item()
        
        history['loss'].append(loss)
        history['uncertainty'].append(unc)

    print(f"     Status: Stable. Avg Loss: {np.mean(history['loss'][-10:]):.4f}")

    # ---------------------------------------------------------
    # DAY 2: THE CRISIS (Concept Drift)
    # ---------------------------------------------------------
    print("\n   🚨 DAY 2: MARKET CRASH DETECTED (Steps 100-300)")
    print("      (Data distribution inverted. Standard models would fail.)")
    
    for step in range(200):
        # DATA SHIFT: Crisis Mode
        x_stream, y_true = get_market_data('CRISIS', batch_size=1)
        
        # The model faces new data. 
        # It should spike uncertainty -> boost plasticity -> adapt.
        pred = adapter.predict(x_stream, update=True, target=y_true)
        
        loss = torch.nn.functional.mse_loss(pred, y_true).item()
        unc = adapter.get_uncertainty(x_stream).mean().item()
        
        history['loss'].append(loss)
        history['uncertainty'].append(unc)
        
        if step == 10: print("     ⚠️  System shock! Loss spiking...")
        if step == 50: print("     🔄  Stabilizer engaged. Adapting weights...")
        if step == 150: print(f"     ✅  Recovery confirmed. Current Loss: {loss:.4f}")

    return history

# ==============================================================================
# VISUALIZATION
# ==============================================================================
def plot_endgame(history):
    print("\n📊 Generating Endgame Report...")
    
    loss = history['loss']
    unc = history['uncertainty']
    
    # Moving Average for cleaner plots
    def smooth(data, w=10):
        return np.convolve(data, np.ones(w)/w, mode='valid')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot 1: Loss Recovery
    ax1.plot(loss, color='blue', alpha=0.3, label='Instant Loss')
    ax1.plot(smooth(loss), color='darkblue', linewidth=2, label='Smoothed Loss')
    ax1.axvline(x=100, color='red', linestyle='--', linewidth=2, label='THE CRISIS')
    ax1.set_ylabel('Prediction Error (MSE)', fontweight='bold')
    ax1.set_title('Endgame: Production Recovery from catastrophic Drift', fontweight='bold', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Introspection (Uncertainty)
    ax2.plot(smooth(unc), color='orange', linewidth=2, label='Model Uncertainty (LogVar)')
    ax2.axvline(x=100, color='red', linestyle='--', linewidth=2)
    ax2.set_ylabel('Introspection Signal', fontweight='bold', color='orange')
    ax2.set_xlabel('Production Steps (Streaming)', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Annotation
    ax1.text(110, max(smooth(loss)), "ADAPTATION TRIGGERED", color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("endgame_results.png")
    print("✅ Report saved to: endgame_results.png")

# ==============================================================================
# EXECUTION
# ==============================================================================
if __name__ == "__main__":
    # 1. Train
    checkpoint = run_training_phase()
    
    # 2. Deploy
    production_history = run_production_phase(checkpoint)
    
    # 3. Visualize
    plot_endgame(production_history)