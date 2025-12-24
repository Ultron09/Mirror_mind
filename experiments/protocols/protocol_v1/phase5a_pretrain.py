"""
PHASE 5A: ARC PRE-TRAINING (THE SCHOOL) - WITH REPORTING & SYNTHETIC FALLBACK
=============================================================================
Goal: Implant "Priors" (Objectness, Gravity, Symmetry) into the model.
Method: Meta-Learning (Reptile) across Real or Synthetic tasks.
Updates:
1. Synthetic Data Generator (Fallback if no real data).
2. Corrected Reptile Update Logic (Theta <- Theta + Epsilon * (Theta_Prime - Theta)).
3. Automated Visualization & Reporting.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import glob
import os
import random
import logging
import sys
import copy
import platform
import datetime
import numpy as np
import matplotlib.pyplot as plt

# Try to import tqdm, fallback if missing
try:
    from tqdm import tqdm
except ImportError:
    # Mock tqdm if not installed
    def tqdm(iterable, desc=""):
        print(f"   {desc}...")
        return iterable

# Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from airbornehrs import (
        AdaptiveFramework, 
        AdaptiveFrameworkConfig, 
        MetaController,
        MetaControllerConfig
    )
except ImportError:
    print("❌ Critical: airbornehrs package not found. (Mocking for standalone test)")
    # Simple Mock for standalone testing
    class AdaptiveFramework: pass
    class AdaptiveFrameworkConfig: pass 
    class MetaController: pass
    class MetaControllerConfig: pass

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("PreTrain")

# ==============================================================================
# HELPER: Visualization & Reporting
# ==============================================================================
def generate_artifacts(loss_history, config, status):
    """Generates PNG plot and MD report for research documentation."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # --- 1. Generate Visualization (PNG) ---
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, marker='o', linestyle='-', color='#8e44ad', label='Meta-Loss')
        plt.title(f"MirrorMind Pre-Training: Prior Acquisition\n{timestamp}")
        plt.xlabel("Epoch")
        plt.ylabel("Avg Task Loss")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.savefig("pretrain_loss_curve.png", dpi=300)
        plt.close()
        logger.info("   ✅ Visualization saved: pretrain_loss_curve.png")
    except Exception as e:
        logger.error(f"   ⚠️ Visualization failed: {e}")

    # --- 2. Generate Research Report (Markdown) ---
    report_content = f"""# MirrorMind Protocol: Phase 5A Pre-Training Report
**Date:** {timestamp}
**Status:** {status}
**Mode:** {config['mode']}

## 1. Executive Summary
The system underwent meta-training to acquire inductive priors suitable for grid reasoning tasks.
* **Tasks Processed:** {config['num_tasks']}
* **Epochs:** {config['epochs']}
* **Final Loss:** {loss_history[-1] if loss_history else 'N/A'}

## 2. Configuration
* **Algorithm:** Reptile (First-Order MAML)
* **Inner Steps:** {config['inner_steps']}
* **Meta-Learning Rate (Epsilon):** {config['epsilon']}
* **Device:** {config['device']}

## 3. Convergence
The loss trajectory indicates {"successful acquisition of priors" if loss_history[-1] < 1.0 else "underfitting/instability"}. 
(See pretrain_loss_curve.png for details).

## 4. Conclusion
The 'checkpoints/arc_pretrained_final.pt' file now contains the "educated" weights. These weights should be used as the initialization for Phase 5 (The Exam).
"""
    
    with open("PRETRAIN_REPORT.md", "w") as f:
        f.write(report_content)
    logger.info("   ✅ Research Report generated: PRETRAIN_REPORT.md")

# ==============================================================================
# 1. SYNTHETIC DATA GENERATOR (Fallback)
# ==============================================================================
class SyntheticARC:
    """Generates procedural ARC-like tasks if real data is missing."""
    @staticmethod
    def generate_task(task_type=None):
        if task_type is None:
            task_type = random.choice(['color_shift', 'vertical_flip', 'denoise'])
            
        pairs = []
        # Generate 5 training pairs
        for _ in range(5):
            inp, outp = SyntheticARC._make_pair(task_type)
            pairs.append({'input': inp, 'output': outp})
        return pairs

    @staticmethod
    def _make_pair(task_type):
        h, w = random.randint(3, 15), random.randint(3, 15)
        grid = np.random.randint(0, 10, (h, w))
        
        if task_type == 'color_shift':
            # Rule: Add 1 to color index (wrapping at 10)
            out_grid = (grid + 1) % 10
        elif task_type == 'vertical_flip':
            # Rule: Flip upside down
            out_grid = np.flipud(grid)
        elif task_type == 'denoise':
            # Rule: Remove noise (color 5), replace with black (0)
            out_grid = grid.copy()
            out_grid[out_grid == 5] = 0
            
        return grid.tolist(), out_grid.tolist()

# ==============================================================================
# 2. MODEL ARCHITECTURE
# ==============================================================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
    def forward(self, x):
        return x + self.conv(x)

class ResNetGridReasoner(nn.Module):
    def __init__(self):
        super().__init__()
        self.entry = nn.Conv2d(10, 64, 3, padding=1)
        self.body = nn.Sequential(ResBlock(64), ResBlock(64), ResBlock(64), ResBlock(64))
        self.exit = nn.Conv2d(64, 10, 1) # Logits out
        
    def forward(self, x):
        x = self.entry(x)
        x = self.body(x)
        return self.exit(x)

# ==============================================================================
# 3. DATA & UTILS
# ==============================================================================
def load_tasks(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.json"))
    if not files:
        return None
    
    tasks = []
    logger.info(f"   📂 Loading {len(files)} tasks from disk...")
    for f_path in files:
        with open(f_path, 'r') as f:
            data = json.load(f)
            tasks.append(data['train']) 
    return tasks

def grid_to_tensor(grid, max_h=30, max_w=30):
    t = torch.zeros(10, max_h, max_w)
    h = min(len(grid), max_h)
    w = min(len(grid[0]), max_w)
    
    for r in range(h):
        for c in range(w):
            val = int(grid[r][c])
            if 0 <= val < 10:
                t[val, r, c] = 1.0
            
    target = torch.zeros(max_h, max_w, dtype=torch.long)
    for r in range(h):
        for c in range(w):
            val = int(grid[r][c])
            if 0 <= val < 10:
                target[r, c] = val
            
    return t.unsqueeze(0), target.unsqueeze(0)

# ==============================================================================
# 4. PRE-TRAINING LOOP (REPTILE)
# ==============================================================================
def run_pretraining():
    logger.info("🏫 PHASE 5A: STARTING PRE-TRAINING SCHOOL")
    
    # 1. Config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"   🚀 Accelerator: {device}")
    
    # 2. Data Loading (Real vs Synthetic)
    tasks = load_tasks("data/training")
    mode = "Real Data"
    
    if not tasks:
        logger.warning("   ⚠️ No data found in data/training. Using SYNTHETIC engine.")
        tasks = [SyntheticARC.generate_task() for _ in range(100)] # Generate 100 fake tasks
        mode = "Synthetic Fallback"
    
    # Ensure checkpoints dir
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
        
    # 3. Model Setup
    meta_model = ResNetGridReasoner().to(device)
    meta_optimizer = optim.AdamW(meta_model.parameters(), lr=0.001) # For inner loop
    criterion = nn.CrossEntropyLoss()
    
    # Reptile Hyperparams
    EPOCHS = 10
    INNER_STEPS = 5
    META_EPSILON = 0.1 # Step size towards new weights
    
    loss_history = []
    
    logger.info(f"   🔄 Commencing Curriculum ({mode})...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        random.shuffle(tasks)
        
        # Save Meta-Weights (Phi) before epoch starts or update incrementally?
        # Reptile Standard: For each task:
        # 1. Phi_old = current_weights
        # 2. Train k steps -> Phi_new
        # 3. current_weights += epsilon * (Phi_new - Phi_old)
        
        progress = tqdm(tasks, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for task_pairs in progress:
            # A. Snapshot Meta-Weights
            meta_weights = copy.deepcopy(meta_model.state_dict())
            
            # B. Inner Loop (Train on specific task)
            task_loss_accum = 0
            for _ in range(INNER_STEPS):
                # Sample a random pair from the task to create a batch (or use all)
                pair = random.choice(task_pairs)
                x, y = grid_to_tensor(pair['input'])
                x, y = x.to(device), y.to(device)
                
                meta_optimizer.zero_grad()
                pred_logits = meta_model(x)
                loss = criterion(pred_logits, y)
                loss.backward()
                meta_optimizer.step()
                
                task_loss_accum += loss.item()
            
            avg_task_loss = task_loss_accum / INNER_STEPS
            total_loss += avg_task_loss
            
            # C. Reptile Update (Meta-Optimization)
            # Phi = Phi + Epsilon * (Theta_Prime - Phi)
            current_weights = meta_model.state_dict() # This is Theta_Prime
            new_meta_weights = {}
            
            with torch.no_grad():
                for name in meta_weights:
                    # diff = Trained - Old
                    diff = current_weights[name] - meta_weights[name]
                    # Update = Old + Epsilon * diff
                    new_meta_weights[name] = meta_weights[name] + (META_EPSILON * diff)
            
            # Apply updated weights back to model for next task
            meta_model.load_state_dict(new_meta_weights)
            
            progress.set_postfix({'loss': avg_task_loss})

        avg_epoch_loss = total_loss / len(tasks)
        loss_history.append(avg_epoch_loss)
        logger.info(f"   📉 Epoch {epoch+1} Complete. Avg Loss: {avg_epoch_loss:.4f}")
        
        # Save Milestone
        if (epoch + 1) % 5 == 0:
            torch.save(meta_model.state_dict(), f"checkpoints/arc_pretrained_epoch_{epoch+1}.pt")

    # Final Save
    logger.info("   ✅ Pre-Training Complete. Brain Saved.")
    torch.save(meta_model.state_dict(), "checkpoints/arc_pretrained_final.pt")
    
    # Artifacts
    config_dump = {
        'mode': mode,
        'num_tasks': len(tasks),
        'epochs': EPOCHS,
        'inner_steps': INNER_STEPS,
        'epsilon': META_EPSILON,
        'device': device
    }
    generate_artifacts(loss_history, config_dump, status="COMPLETED")

if __name__ == "__main__":
    run_pretraining()