"""
PHASE 5A: ARC PRE-TRAINING (THE SCHOOL)
=======================================
Goal: Implant "Priors" (Objectness, Gravity, Symmetry) into the model 
so it doesn't have to learn them from scratch during the exam.

Dataset: 400 Training Tasks from 'data/training/'
Method: Meta-Learning (Reptile) across all 400 tasks.
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
import numpy as np
from tqdm import tqdm

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
    print("❌ Critical: airbornehrs package not found.")
    sys.exit(1)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("PreTrain")

# ==============================================================================
# 1. MODEL ARCHITECTURE (ResNet U-Net)
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
        # 10 Colors -> 64 Features
        self.entry = nn.Conv2d(10, 64, 3, padding=1)
        self.body = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
            ResBlock(64),
            ResBlock(64) # Deeper for pre-training
        )
        # 64 Features -> 10 Colors
        self.exit = nn.Conv2d(64, 10, 1)
        # No Softmax here, CrossEntropyLoss expects raw logits
        
    def forward(self, x):
        x = self.entry(x)
        x = self.body(x)
        return self.exit(x)

# ==============================================================================
# 2. DATA LOADER
# ==============================================================================
def load_all_tasks(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.json"))
    tasks = []
    
    logger.info(f"   📂 Loading {len(files)} tasks from disk...")
    
    for f_path in files:
        with open(f_path, 'r') as f:
            data = json.load(f)
            tasks.append(data['train']) # Only use 'train' pairs for pre-training
            
    return tasks

def grid_to_tensor(grid, max_h=30, max_w=30):
    # Standardize size with padding (0 = background/black)
    t = torch.zeros(10, max_h, max_w) # [C, H, W]
    h = min(len(grid), max_h)
    w = min(len(grid[0]), max_w)
    
    for r in range(h):
        for c in range(w):
            val = int(grid[r][c])
            t[val, r, c] = 1.0 # One-hot-ish input
            
    # Target: Class Indices [H, W]
    target = torch.zeros(max_h, max_w, dtype=torch.long)
    for r in range(h):
        for c in range(w):
            target[r, c] = int(grid[r][c])
            
    return t.unsqueeze(0), target.unsqueeze(0) # Add Batch Dim

# ==============================================================================
# 3. PRE-TRAINING LOOP
# ==============================================================================
def run_pretraining():
    logger.info("🏫 PHASE 5A: STARTING PRE-TRAINING SCHOOL")
    
    # 1. Config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"   🚀 Accelerator: {device}")
    
    tasks = load_all_tasks("data/training")
    if not tasks:
        logger.error("   ❌ No data found! Run 'python arc_data.py' first.")
        sys.exit(1)
        
    fw_config = AdaptiveFrameworkConfig(
        model_dim=64,
        learning_rate=0.001,
        compile_model=False,
        device=device
    )
    
    model = ResNetGridReasoner()
    framework = AdaptiveFramework(model, fw_config)
    
    # Meta-Controller for global stability
    meta_config = MetaControllerConfig(use_reptile=True, reptile_update_interval=5)
    controller = MetaController(framework, meta_config)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    EPOCHS = 5  # Passes over the full 400 tasks
    
    logger.info("   🔄 Commencing Curriculum...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        random.shuffle(tasks)
        
        progress = tqdm(tasks, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for task_pairs in progress:
            # For each task, we train on its pairs
            task_loss = 0
            
            # Reptile Inner Loop (Task Specific)
            fast_weights = {n: p.clone() for n, p in model.named_parameters()}
            
            for pair in task_pairs:
                x, y = grid_to_tensor(pair['input'])
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                pred_logits = model(x) # [B, 10, H, W]
                
                loss = criterion(pred_logits, y)
                loss.backward()
                optimizer.step()
                
                task_loss += loss.item()
                
            total_loss += (task_loss / len(task_pairs))
            progress.set_postfix({'loss': task_loss / len(task_pairs)})
            
            # Reptile Meta-Update (Slow Weights)
            # Pull slow weights slightly towards the task solution
            with torch.no_grad():
                epsilon = 0.1
                for name, param in model.named_parameters():
                    param.data = param.data + epsilon * (fast_weights[name] - param.data)

        avg_epoch_loss = total_loss / len(tasks)
        logger.info(f"   📉 Epoch {epoch+1} Complete. Avg Loss: {avg_epoch_loss:.4f}")
        
        # Save Milestone
        torch.save(model.state_dict(), f"checkpoints/arc_pretrained_epoch_{epoch+1}.pt")

    logger.info("   ✅ Pre-Training Complete. Brain Saved.")
    torch.save(model.state_dict(), "checkpoints/arc_pretrained_final.pt")

if __name__ == "__main__":
    run_pretraining()