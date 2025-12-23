"""
PHASE 5A (V2): ARC PRE-TRAINING - MIRRORMIND "THE BEAST" ARCHITECTURE
=====================================================================
Goal: Implant robust "Priors" using a Hybrid Conv-Transformer Architecture.
Model: MirrorMindV2 (Attention + Convolution + Positional Awareness).
Method: Meta-Learning (Reptile) with Gradient Clipping & Gelu Activations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import glob
import os
import random
import logging
import sys
import copy
import math
import platform
import datetime
import numpy as np
import matplotlib.pyplot as plt

# Try to import tqdm
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        print(f"   {desc}...")
        return iterable

# Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("PreTrain_V2")

# ==============================================================================
# 1. THE BEAST: MIRRORMIND V2 ARCHITECTURE
# ==============================================================================

class PositionalEncoding2D(nn.Module):
    """
    Injects 2D spatial awareness so the Attention mechanism knows 
    'Top-Left' from 'Bottom-Right'. Crucial for Symmetry/Gravity tasks.
    """
    def __init__(self, channels, max_h=30, max_w=30):
        super().__init__()
        self.row_embed = nn.Embedding(max_h, channels // 2)
        self.col_embed = nn.Embedding(max_w, channels // 2)
        
    def forward(self, x):
        # x: [Batch, Channels, H, W]
        b, c, h, w = x.shape
        
        # Create grids
        y_coords = torch.arange(h, device=x.device).unsqueeze(1).expand(h, w)
        x_coords = torch.arange(w, device=x.device).unsqueeze(0).expand(h, w)
        
        # Lookup embeddings
        y_emb = self.row_embed(y_coords) # [H, W, C/2]
        x_emb = self.col_embed(x_coords) # [H, W, C/2]
        
        # Concatenate to form full channel dimension
        pos = torch.cat([y_emb, x_emb], dim=-1) # [H, W, C]
        pos = pos.permute(2, 0, 1).unsqueeze(0).expand(b, -1, -1, -1) # [B, C, H, W]
        
        return x + pos

class ConvBlock(nn.Module):
    """Deep Local Reasoning (ResNet-style with Modern Tweaks)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(), # Modern activation
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        # 1x1 conv for skip connection if dimensions change
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return F.gelu(self.body(x) + self.skip(x))

class TransformerBlock(nn.Module):
    """Global Reasoning (Self-Attention)"""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )

    def forward(self, x):
        # Input: [B, C, H, W]
        b, c, h, w = x.shape
        
        # Flatten for Transformer: [B, H*W, C]
        flat = x.view(b, c, -1).permute(0, 2, 1)
        
        # 1. Self-Attention (Global Context)
        res = flat
        flat = self.norm1(flat)
        attn_out, _ = self.attn(flat, flat, flat)
        flat = res + attn_out
        
        # 2. Feed Forward (Reasoning)
        res = flat
        flat = self.norm2(flat)
        mlp_out = self.mlp(flat)
        flat = res + mlp_out
        
        # Reshape back: [B, C, H, W]
        return flat.permute(0, 2, 1).view(b, c, h, w)

class MirrorMindV2(nn.Module):
    """
    The Beast. 
    Combines ConvNet (Seeing) + Transformer (Thinking) + Positional Encoding (Orienting).
    """
    def __init__(self):
        super().__init__()
        self.dim = 128 # High capacity
        
        # 1. Perceptual Stem
        self.stem = nn.Sequential(
            nn.Conv2d(10, self.dim, 3, padding=1),
            nn.GELU()
        )
        self.pos_encoder = PositionalEncoding2D(self.dim)
        
        # 2. Encoder (Visual Extraction)
        self.enc1 = ConvBlock(self.dim, self.dim)
        self.enc2 = ConvBlock(self.dim, self.dim)
        
        # 3. The Core (Global Reasoning / MLA)
        # Allows pixels to "talk" to distant pixels (Teleportation logic)
        self.transformer = nn.Sequential(
            TransformerBlock(self.dim, num_heads=8),
            TransformerBlock(self.dim, num_heads=8),
            TransformerBlock(self.dim, num_heads=8) 
        )
        
        # 4. Decoder (Refinement)
        self.dec1 = ConvBlock(self.dim, self.dim)
        self.dec2 = ConvBlock(self.dim, self.dim)
        
        # 5. Output Head
        self.head = nn.Conv2d(self.dim, 10, 1) # Map back to 10 colors
        
    def forward(self, x):
        # x: [B, 10, H, W]
        
        # Embed & Position
        x = self.stem(x)
        x = self.pos_encoder(x)
        
        # Encode (Save residuals for potential U-Net expansion later)
        x = self.enc1(x)
        x = self.enc2(x)
        
        # Global Reasoning (The Aha! Moment)
        x = self.transformer(x)
        
        # Decode
        x = self.dec1(x)
        x = self.dec2(x)
        
        # Logits
        return self.head(x)

# ==============================================================================
# 2. SYNTHETIC DATA GENERATOR
# ==============================================================================
class SyntheticARC:
    """Generates procedural ARC-like tasks."""
    @staticmethod
    def generate_task():
        task_type = random.choice(['color_shift', 'vertical_flip', 'denoise', 'rect_fill'])
        pairs = []
        for _ in range(5):
            inp, outp = SyntheticARC._make_pair(task_type)
            pairs.append({'input': inp, 'output': outp})
        return pairs

    @staticmethod
    def _make_pair(task_type):
        h, w = random.randint(5, 20), random.randint(5, 20)
        grid = np.random.randint(0, 10, (h, w))
        
        if task_type == 'color_shift':
            out_grid = (grid + 1) % 10
        elif task_type == 'vertical_flip':
            out_grid = np.flipud(grid)
        elif task_type == 'denoise':
            out_grid = grid.copy()
            noise_mask = np.random.random(grid.shape) > 0.8
            grid[noise_mask] = 5 # Add grey noise
            out_grid = grid.copy()
            out_grid[out_grid == 5] = 0 # Clean it
        elif task_type == 'rect_fill':
            # Create a black grid with a single colored rectangle
            grid = np.zeros((h, w), dtype=int)
            out_grid = np.zeros((h, w), dtype=int)
            if h > 4 and w > 4:
                rx, ry = random.randint(0, w-4), random.randint(0, h-4)
                rh, rw = random.randint(2, 4), random.randint(2, 4)
                color = random.randint(1, 9)
                # Input: Outline
                grid[ry:ry+rh, rx:rx+rw] = color 
                grid[ry+1:ry+rh-1, rx+1:rx+rw-1] = 0
                # Output: Filled
                out_grid[ry:ry+rh, rx:rx+rw] = color
        
        if 'out_grid' not in locals(): out_grid = grid # Fallback
        return grid.tolist(), out_grid.tolist()

# ==============================================================================
# 3. UTILS & DATA
# ==============================================================================
def load_tasks(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.json"))
    if not files: return None
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
            if 0 <= val < 10: t[val, r, c] = 1.0
    target = torch.zeros(max_h, max_w, dtype=torch.long)
    for r in range(h):
        for c in range(w):
            val = int(grid[r][c])
            if 0 <= val < 10: target[r, c] = val
    return t.unsqueeze(0), target.unsqueeze(0)

def generate_artifacts(loss_history, config, status):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, marker='o', linestyle='-', color='#e74c3c', label='MirrorMind V2 Loss')
        plt.title(f"MirrorMind V2 Pre-Training Report\n{timestamp}")
        plt.xlabel("Epoch")
        plt.ylabel("Avg Meta-Loss")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.savefig("pretrain_v2_loss.png", dpi=300)
        plt.close()
    except Exception: pass
    
    report = f"""# MirrorMind V2 (The Beast) Report
**Date:** {timestamp}
**Architecture:** Hybrid Conv-Transformer (MLA)

## Executive Summary
* **Final Loss:** {loss_history[-1] if loss_history else 'N/A'}
* **Epochs:** {config['epochs']}
* **Mode:** {config['mode']}

## Architecture Details
* **Dim:** 128 Channels
* **Attention:** 3 Layers x 8 Heads
* **Positional Encoding:** Learned 2D Embeddings
"""
    with open("PRETRAIN_V2_REPORT.md", "w") as f:
        f.write(report)
    logger.info("   ✅ V2 Report Generated.")

# ==============================================================================
# 4. TRAINING LOOP
# ==============================================================================
def run_pretraining_v2():
    logger.info("🦁 PHASE 5A (V2): UNLEASHING THE BEAST")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"   🚀 Accelerator: {device}")
    
    # Load Data
    tasks = load_tasks("data/training")
    mode = "Real Data"
    if not tasks:
        logger.warning("   ⚠️ No data found. Using SYNTHETIC engine.")
        tasks = [SyntheticARC.generate_task() for _ in range(200)] # More tasks for the beast
        mode = "Synthetic Fallback"

    if not os.path.exists("checkpoints"): os.makedirs("checkpoints")
        
    # Initialize The Beast
    meta_model = MirrorMindV2().to(device)
    num_params = sum(p.numel() for p in meta_model.parameters())
    logger.info(f"   🧠 Model Parameters: {num_params:,}")
    
    meta_optimizer = optim.AdamW(meta_model.parameters(), lr=0.0005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    EPOCHS = 15 # More epochs for larger model
    INNER_STEPS = 5
    META_EPSILON = 0.1 
    
    loss_history = []
    
    for epoch in range(EPOCHS):
        total_loss = 0
        random.shuffle(tasks)
        progress = tqdm(tasks, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for task_pairs in progress:
            meta_weights = copy.deepcopy(meta_model.state_dict())
            
            task_loss_accum = 0
            for _ in range(INNER_STEPS):
                pair = random.choice(task_pairs)
                x, y = grid_to_tensor(pair['input'])
                x, y = x.to(device), y.to(device)
                
                meta_optimizer.zero_grad()
                pred_logits = meta_model(x)
                loss = criterion(pred_logits, y)
                loss.backward()
                
                # Gradient Clipping for Transformer stability
                torch.nn.utils.clip_grad_norm_(meta_model.parameters(), 1.0)
                
                meta_optimizer.step()
                task_loss_accum += loss.item()
            
            avg_task_loss = task_loss_accum / INNER_STEPS
            total_loss += avg_task_loss
            
            # Reptile Update
            current_weights = meta_model.state_dict()
            new_meta_weights = {}
            with torch.no_grad():
                for name in meta_weights:
                    diff = current_weights[name] - meta_weights[name]
                    new_meta_weights[name] = meta_weights[name] + (META_EPSILON * diff)
            meta_model.load_state_dict(new_meta_weights)
            
            progress.set_postfix({'loss': avg_task_loss})

        avg_epoch_loss = total_loss / len(tasks)
        loss_history.append(avg_epoch_loss)
        logger.info(f"   📉 Epoch {epoch+1} Loss: {avg_epoch_loss:.4f}")
        
        if (epoch + 1) % 5 == 0:
            torch.save(meta_model.state_dict(), f"checkpoints/arc_v2_epoch_{epoch+1}.pt")

    logger.info("   ✅ Pre-Training V2 Complete.")
    torch.save(meta_model.state_dict(), "checkpoints/arc_pretrained_final.pt")
    
    config_dump = {'mode': mode, 'epochs': EPOCHS, 'epsilon': META_EPSILON}
    generate_artifacts(loss_history, config_dump, status="COMPLETED")

if __name__ == "__main__":
    run_pretraining_v2()