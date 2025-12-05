import argparse
import json
import os
import random
import datetime
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# IMPORTS: Switching to direct package imports to ensure we use the installed library.
# This validates the 'airbornehrs' package integrity.
try:
    from airbornehrs import (
        AdaptiveFramework,
        AdaptiveFrameworkConfig,
        MetaController
    )
    from airbornehrs.core import PerformanceMonitor # Accessing internal monitor for advanced metrics
except ImportError:
    print("❌ Error: 'airbornehrs' package not found.")
    print("Please install it via 'pip install -e.' from the project root.")
    sys.exit(1)

def set_seed(seed: int):
    """Sets random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(path: str):
    """Loads experiment configuration from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def make_synthetic(data_cfg):
    """Generates synthetic sequence data for regression tasks."""
    n = data_cfg.get("num_samples", 256)
    seq = data_cfg.get("seq_len", 10)
    feat = data_cfg.get("feature_dim", 128)
    # Generate random input and target tensors
    X = torch.randn(n, seq, feat)
    y = torch.randn(n, seq, feat)
    return X, y

def run(config_path: str, output_dir: str = None):
    # 1. Configuration & Initialization
    raw_cfg = load_config(config_path)
    seed = raw_cfg.get("seed", 42)
    set_seed(seed)
    
    device = raw_cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA requested but not available; switching to CPU")
        device = "cpu"

    print(f"\n🚀 Initializing MirrorMind Experiment on {device.upper()}...")
    print(f"📄 Configuration: {os.path.basename(config_path)}")

    # 2. Data Preparation
    # We explicitly handle data loading here to allow for future expansion 
    # into non-synthetic datasets, satisfying the "cool experiments" requirement.
    data_cfg = raw_cfg.get("data", {})
    if data_cfg.get("use_synthetic", True):
        print("🧪 Generating synthetic data...")
        X, y = make_synthetic(data_cfg)
    else:
        raise RuntimeError("Non-synthetic data loading not implemented.")

    # Create DataLoaders
    train_cfg = raw_cfg.get("training", {})
    batch_size = train_cfg.get("batch_size", 32)
    epochs = train_cfg.get("epochs", 3)
    
    # 80/20 Train/Validation Split
    split_idx = int(len(X) * 0.8)
    train_ds = TensorDataset(X[:split_idx], y[:split_idx])
    val_ds = TensorDataset(X[split_idx:], y[split_idx:])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # 3. Framework Instantiation (The "Gun" & "Stabilizer")
    # Mapping raw config to the package's strong-typed configuration class
    framework_config = AdaptiveFrameworkConfig(
        model_dim=raw_cfg.get("model_dim", 128),
        num_layers=raw_cfg.get("num_layers", 4),
        num_heads=raw_cfg.get("num_heads", 4),
        learning_rate=train_cfg.get("learning_rate", 1e-3),
        weight_adaptation_lr=train_cfg.get("weight_adaptation_lr", 1e-5),
        batch_size=batch_size,
        epochs=epochs
    )
    
    # Initialize the core model
    framework = AdaptiveFramework(framework_config, device=device)
    
    # Initialize the MetaController (The "Cool" Part)
    # This component monitors gradients and adjusts LR dynamically
    meta_controller = MetaController(framework)
    
    print("✅ Framework & MetaController initialized.")
    print("⚡ Starting Optimization Cycle...")

    # 4. The Optimization Cycle (Explicit Training Loop)
    history = []
    
    for epoch in range(epochs):
        framework.model.train()
        epoch_loss = 0.0
        adaptation_stats = []
        
        # --- Training Phase ---
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # A. Standard Training Step (Forward/Backward/Optimizer)
            # The framework handles introspection internally during this step
            metrics = framework.train_step(batch_X, batch_y)
            epoch_loss += metrics['loss']
            
            # B. Meta-Adaptation Step (The "Stabilizer" Logic)
            # We actively invoke the meta-controller to inspect the gradients 
            # and potentially adjust the learning rate or regularization.
            meta_stats = meta_controller.adapt(
                loss=metrics['loss'],
                # Gradients are accessible via framework.model.parameters() inside the controller
            )
            adaptation_stats.append(meta_stats)
            
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Calculate average meta-metrics for the epoch
        avg_lr = np.mean([s.get('learning_rate', 0.0) for s in adaptation_stats])
        avg_diff = np.mean([s.get('curriculum_difficulty', 0.0) for s in adaptation_stats])
        
        # --- Validation Phase ---
        framework.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                pred, uncertainty = framework.forward(batch_X) 
                loss = torch.nn.functional.mse_loss(pred, batch_y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        # --- "Cool" Reporting ---
        # Instead of just loss, we report the dynamic adjustments made by the framework
        print(f"Epoch {epoch+1}/{epochs} "
              f"| Train Loss: {avg_train_loss:.4f} "
              f"| Val Loss: {avg_val_loss:.4f} "
              f"| LR: {avg_lr:.2e} "  # Shows dynamic LR adjustment
              f"| Difficulty: {avg_diff:.2f}") # Shows curriculum progress
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "avg_lr": avg_lr,
            "avg_difficulty": avg_diff,
            "meta_stats": adaptation_stats[-1] # Snapshot of final batch stats
        })

    # 5. Summary & Serialization
    summary = {
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "total_epochs": epochs,
        "history": history
    }

    out = {
        "experiment_name": raw_cfg.get("experiment_name", "unnamed"),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "config_path": os.path.abspath(config_path),
        "summary": summary
    }

    if output_dir is None:
        output_dir = os.path.join("runs", raw_cfg.get("experiment_name", "run"))
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "summary.json")
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        
    print(f"\n✅ Experiment complete.")
    print(f"📊 Summary saved to: {out_path}")
    print("💡 Analysis: Check 'summary.json' to see how the Learning Rate evolved!")
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run AirborneHRS Self-Learning Experiment")
    p.add_argument("--config", required=True, help="Path to JSON config file")
    p.add_argument("--output_dir", default=None, help="Directory to save output artifacts")
    args = p.parse_args()
    run(args.config, args.output_dir)