"""
PROTOCOL PHASE 5 (V4): PRE-TRAINED ARC CHALLENGE
================================================
Goal: Prove MirrorMind adapts faster when initialized with priors.
Changes:
1. LOADS 'checkpoints/arc_pretrained_final.pt' (The Educated Brain).
2. Uses standard TTT (Test-Time Training) to fine-tune on the specific task.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import logging
import sys
import os
import numpy as np
import copy
import matplotlib.pyplot as plt
import json
import glob
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig, ProductionAdapter, InferenceMode
except ImportError:
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger("Phase5_V4")

# --- MODEL (MUST MATCH PRE-TRAIN) ---
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
        self.exit = nn.Conv2d(64, 10, 1)
        self.softmax = nn.Softmax(dim=1) # Inference needs probabilities
        
    def forward(self, x):
        x = self.entry(x)
        x = self.body(x)
        x = self.exit(x)
        return self.softmax(x)

# --- DATA LOADER (Validation Split) ---
def load_eval_task(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.json"))
    if not files: return None
    choice = random.choice(files)
    with open(choice, 'r') as f: data = json.load(f)
    
    def grid_to_tensor(grid_list):
        t = torch.zeros(1, 10, 30, 30)
        h, w = min(len(grid_list), 30), min(len(grid_list[0]), 30)
        for r in range(h):
            for c in range(w):
                t[0, grid_list[r][c], r, c] = 1.0
        return t

    sx, sy = [], []
    for pair in data['train']: # Use training examples as Support Set
        sx.append(grid_to_tensor(pair['input']))
        sy.append(grid_to_tensor(pair['output']))
        
    qx, qy = [], []
    for pair in data['test']: # Use test examples as Query Set
        qx.append(grid_to_tensor(pair['input']))
        qy.append(grid_to_tensor(pair['output']))
        
    return torch.cat(sx), torch.cat(sy), torch.cat(qx), torch.cat(qy), os.path.basename(choice)

# --- BENCHMARK ---
def run_arc_v4():
    logger.info("🏆 PHASE 5 (V4): PRE-TRAINED ARC CHALLENGE")
    
    # 1. Load Pre-Trained Weights
    PRETRAIN_PATH = "checkpoints/arc_pretrained_final.pt"
    if not os.path.exists(PRETRAIN_PATH):
        logger.error("❌ Pre-trained weights not found! Run 'phase5a_pretrain.py' first.")
        sys.exit(1)
        
    logger.info(f"   🧠 Loading Meta-Brain: {PRETRAIN_PATH}")
    
    # 2. Setup
    NUM_TASKS = 50
    ADAPT_STEPS = 10 # Allow more steps since we have priors
    score_mm, score_base = 0.0, 0.0
    improvements = []
    
    for i in range(NUM_TASKS):
        task_data = load_eval_task("data/training")
        if not task_data: break
        sx, sy, qx, qy, name = task_data
        
        # Init Models (Both start with Pre-Trained Knowledge)
        base_model = ResNetGridReasoner()
        base_model.load_state_dict(torch.load(PRETRAIN_PATH)) # Baseline = Static Pre-Trained
        
        mm_model = copy.deepcopy(base_model) # MirrorMind = Adaptive Pre-Trained
        
        # --- BASELINE (Static Inference - Zero Shot) ---
        # The baseline represents a standard model that doesn't learn at test time
        base_model.eval()
        with torch.no_grad():
            p_base = base_model(qx)
            # Use MSE for simple scoring comparison
            l_base = nn.MSELoss()(p_base, qy).item()
            
        # --- MIRRORMIND (Test-Time Training) ---
        fw_config = AdaptiveFrameworkConfig(
            learning_rate=0.005, # Higher LR allowed because of priors
            adaptation_threshold=0.01,
            compile_model=False,
            device='cpu'
        )
        framework = AdaptiveFramework(mm_model, fw_config)
        
        # Adaptation Loop
        for _ in range(ADAPT_STEPS):
            framework.train_step(sx, sy, enable_dream=False)
            
        with torch.no_grad():
            p_mm, _, _ = framework.forward(qx)
            l_mm = nn.MSELoss()(p_mm, qy).item()
            
        # --- SCORING ---
        if l_base < 1e-6: l_base = 1e-6
        imp = ((l_base - l_mm) / l_base) * 100.0
        improvements.append(imp)
        
        if l_mm < l_base:
            score_mm += 1
            icon = "✅ MM"
        else:
            score_base += 1
            icon = "❌ Static"
            
        logger.info(f"   Task {i+1:02d} ({name}): Static={l_base:.4f} | Adap={l_mm:.4f} | Δ {imp:+.1f}% | {icon}")

    # Verdict
    avg_imp = np.mean(improvements)
    logger.info("-" * 40)
    logger.info(f"FINAL: MirrorMind {score_mm} - {score_base} Static Baseline")
    logger.info(f"AVG BOOST: {avg_imp:.1f}%")
    
    if avg_imp > 0:
        print("\n" + "="*40)
        print(f"🟢 SUCCESS: Adaptation improved performance by {avg_imp:.1f}%")
        print("="*40 + "\n")
    else:
        print("\n" + "="*40)
        print("🔴 FAILURE: Adaptation caused regression.")
        print("="*40 + "\n")

if __name__ == "__main__":
    run_arc_v4()