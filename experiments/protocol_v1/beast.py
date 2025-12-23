"""
PROTOCOL PHASE 6: ARC BENCHMARK RUNNER (DEPLOYMENT)
===================================================
Goal: Run MirrorMind V2 on the Official Evaluation Set.
Features:
1. "The Beast" Architecture (MirrorMindV2).
2. Test-Time Training (TTT) on every task.
3. Smart Cropping (Infers output size from input logic).
4. Generates submission.json.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import sys
import os
import json
import glob
import numpy as np
import copy
from typing import List

# Setup Paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig
except ImportError:
    print("❌ CRITICAL: airbornehrs package not found.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger("Phase6_Runner")

# ==============================================================================
# 1. THE BEAST (MirrorMind V2)
# ==============================================================================
# Re-defining here to ensure standalone execution reliability
class PositionalEncoding2D(nn.Module):
    def __init__(self, channels, max_h=30, max_w=30):
        super().__init__()
        self.row_embed = nn.Embedding(max_h, channels // 2)
        self.col_embed = nn.Embedding(max_w, channels // 2)
    def forward(self, x):
        b, c, h, w = x.shape
        y_coords = torch.arange(h, device=x.device).unsqueeze(1).expand(h, w)
        x_coords = torch.arange(w, device=x.device).unsqueeze(0).expand(h, w)
        y_emb = self.row_embed(y_coords)
        x_emb = self.col_embed(x_coords)
        pos = torch.cat([y_emb, x_emb], dim=-1).permute(2, 0, 1).unsqueeze(0).expand(b, -1, -1, -1)
        return x + pos

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    def forward(self, x):
        return F.gelu(self.body(x) + self.skip(x))

class TransformerBlock(nn.Module):
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
        b, c, h, w = x.shape
        flat = x.view(b, c, -1).permute(0, 2, 1)
        res = flat
        flat = self.norm1(flat)
        attn_out, _ = self.attn(flat, flat, flat)
        flat = res + attn_out
        res = flat
        flat = self.norm2(flat)
        mlp_out = self.mlp(flat)
        flat = res + mlp_out
        return flat.permute(0, 2, 1).view(b, c, h, w)

class MirrorMindV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 128
        self.stem = nn.Sequential(nn.Conv2d(10, self.dim, 3, padding=1), nn.GELU())
        self.pos_encoder = PositionalEncoding2D(self.dim)
        self.enc1 = ConvBlock(self.dim, self.dim)
        self.enc2 = ConvBlock(self.dim, self.dim)
        self.transformer = nn.Sequential(
            TransformerBlock(self.dim, num_heads=8),
            TransformerBlock(self.dim, num_heads=8),
            TransformerBlock(self.dim, num_heads=8) 
        )
        self.dec1 = ConvBlock(self.dim, self.dim)
        self.dec2 = ConvBlock(self.dim, self.dim)
        self.head = nn.Conv2d(self.dim, 10, 1)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.pos_encoder(x)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.transformer(x)
        x = self.dec1(x)
        x = self.dec2(x)
        return self.head(x)

# ==============================================================================
# 2. SMART CROPPING LOGIC
# ==============================================================================
def detect_size_logic(train_pairs):
    """
    Analyzes training pairs to guess the output size rule.
    Returns: 'same', 'fixed', or 'bbox'
    """
    is_same_size = True
    fixed_size = None
    
    for p in train_pairs:
        h_in, w_in = len(p['input']), len(p['input'][0])
        h_out, w_out = len(p['output']), len(p['output'][0])
        
        if (h_in != h_out) or (w_in != w_out):
            is_same_size = False
        
        if fixed_size is None:
            fixed_size = (h_out, w_out)
        elif fixed_size != (h_out, w_out):
            fixed_size = "variable"
            
    if is_same_size: return 'same', None
    if fixed_size != "variable": return 'fixed', fixed_size
    return 'bbox', None

def tensor_to_grid(t_logits, strategy, input_dims=None, fixed_dims=None):
    """
    Converts 30x30 Logits -> Correctly Sized 2D Grid
    """
    # 1. Get raw prediction (30x30)
    # Use Argmax to get classes 0-9
    pred_full = torch.argmax(t_logits[0], dim=0).cpu().numpy() # [30, 30]
    
    # 2. Crop based on strategy
    if strategy == 'same' and input_dims:
        h, w = input_dims
        return pred_full[:h, :w].tolist()
    
    elif strategy == 'fixed' and fixed_dims:
        h, w = fixed_dims
        return pred_full[:h, :w].tolist()
        
    else: # 'bbox' strategy (Autocrop non-black pixels)
        rows = np.any(pred_full != 0, axis=1)
        cols = np.any(pred_full != 0, axis=0)
        if not np.any(rows) or not np.any(cols):
            return [[0]] # Fallback: Single black pixel
            
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return pred_full[rmin:rmax+1, cmin:cmax+1].tolist()

def grid_to_tensor(grid_list, max_h=30, max_w=30):
    t = torch.zeros(1, 10, max_h, max_w)
    h = min(len(grid_list), max_h)
    w = min(len(grid_list[0]), max_w)
    for r in range(h):
        for c in range(w):
            v = int(grid_list[r][c])
            if 0 <= v < 10: t[0, v, r, c] = 1.0
    return t

# ==============================================================================
# 3. MAIN RUNNER
# ==============================================================================
def run_benchmark():
    logger.info("🚀 PHASE 6: STARTING BENCHMARK RUN")
    
    # Checkpoints
    PRETRAIN_PATH = "checkpoints/arc_pretrained_final.pt"
    if not os.path.exists(PRETRAIN_PATH):
        logger.error("❌ Pre-trained weights missing! Run Phase 5A first.")
        return

    # Load Model
    base_model = MirrorMindV2()
    base_model.load_state_dict(torch.load(PRETRAIN_PATH))
    logger.info(f"🧠 Loaded The Beast from {PRETRAIN_PATH}")

    # Config (Phase 5 V2 Settings)
    fw_config = AdaptiveFrameworkConfig(
        learning_rate=0.005,
        enable_active_shield=True, # Allows model to "Ski" if confident
        active_shield_threshold=0.085,
        active_shield_slope=20.0,
        device='cpu' 
    )

    # Load Tasks
    files = glob.glob("data/evaluation/*.json")
    submission = {}
    
    logger.info(f"📂 Processing {len(files)} tasks...")

    for i, f_path in enumerate(files):
        with open(f_path, 'r') as f: data = json.load(f)
        task_id = os.path.basename(f_path).replace(".json", "")
        
        # 1. Determine Output Size Logic
        size_strategy, fixed_dims = detect_size_logic(data['train'])
        
        # 2. Prepare Support Set (Train)
        sx = torch.cat([grid_to_tensor(p['input']) for p in data['train']])
        sy = torch.cat([grid_to_tensor(p['output']) for p in data['train']])
        
        # 3. Initialize Adaptive Agent
        mm_model = copy.deepcopy(base_model)
        framework = AdaptiveFramework(mm_model, fw_config)
        
        # SAFETY TETHER: Critical for preventing regressions
        framework.ewc.lock_for_ttt(strength=2000.0)
        
        # 4. Test-Time Training (Adaptation)
        framework.model.train()
        for _ in range(10): # 10 Steps
            framework.train_step(sx, sy)
            
        # 5. Inference on Test Set
        framework.model.eval()
        task_preds = []
        
        with torch.no_grad():
            for pair in data['test']:
                qx_raw = pair['input']
                qx = grid_to_tensor(qx_raw)
                
                # Predict
                logits = framework.model(qx)
                
                # Decode & Crop
                input_dims = (len(qx_raw), len(qx_raw[0]))
                pred_grid = tensor_to_grid(logits, size_strategy, input_dims, fixed_dims)
                
                # Format: Each test input needs 2 attempts (we submit same for both)
                task_preds.append({
                    "attempt_1": pred_grid,
                    "attempt_2": pred_grid
                })
        
        submission[task_id] = task_preds
        if (i+1) % 10 == 0: logger.info(f"   Processed {i+1}/{len(files)} tasks...")

    # Save
    with open("submission.json", "w") as f:
        json.dump(submission, f)
    
    logger.info("✅ Benchmark Complete. 'submission.json' generated.")

if __name__ == "__main__":
    run_benchmark()