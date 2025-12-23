"""
PROTOCOL PHASE 5 (STRONG - V2): THE BEAST EVALUATION
====================================================
Goal: Verify MirrorMind V2 (Hybrid MLA) on the ARC Challenge.
Architecture: Conv-Transformer Hybrid (The Beast).
Safety: Active Shield + EWC Tether (2000.0).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import sys
import os
import numpy as np
import copy
import matplotlib.pyplot as plt
import json
import glob
import random
import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig
except ImportError:
    print("❌ CRITICAL: Import failed.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s' , encoding='utf-8') 
logger = logging.getLogger("Phase5_V2_Strong" )

# ==============================================================================
# 1. ARCHITECTURE: MIRRORMIND V2 (Must match Pre-training EXACTLY)
# ==============================================================================

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
    """The Beast Architecture"""
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
# 2. UTILS
# ==============================================================================
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
                v = grid_list[r][c]
                if 0 <= v < 10: t[0, v, r, c] = 1.0
        return t

    sx = torch.cat([grid_to_tensor(p['input']) for p in data['train']])
    sy = torch.cat([grid_to_tensor(p['output']) for p in data['train']])
    qx = torch.cat([grid_to_tensor(p['input']) for p in data['test']])
    qy = torch.cat([grid_to_tensor(p['output']) for p in data['test']])
    return sx, sy, qx, qy, os.path.basename(choice)

def generate_artifacts(results):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Visualization
    try:
        static = [r['static_loss'] for r in results]
        adaptive = [r['adaptive_loss'] for r in results]
        plt.figure(figsize=(8, 8))
        plt.scatter(static, adaptive, c='blue', alpha=0.6, label='Tasks')
        max_val = max(max(static), max(adaptive)) if static else 1.0
        plt.plot([0, max_val], [0, max_val], 'r--', label='No Improvement')
        plt.fill_between([0, max_val], [0, max_val], 0, color='green', alpha=0.1, label='Improvement Zone')
        plt.title(f"MirrorMind V2 (The Beast) Performance\n{timestamp}")
        plt.xlabel("Static Baseline Loss")
        plt.ylabel("Adaptive Loss (Shielded)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig("phase5_v2_performance.png", dpi=300)
        plt.close()
    except Exception: pass

    # 2. Markdown Report
    wins = 0
    skips = 0
    regressions = 0
    
    for r in results:
        if r['plasticity'] < 0.1:
            if r['improvement'] < -1.0: regressions += 1
            else: skips += 1
        elif r['improvement'] > 0:
            wins += 1
        else:
            regressions += 1
            
    avg_imp = np.mean([r['improvement'] for r in results]) if results else 0.0
    
    table_rows = ""
    for i, r in enumerate(results[:20]):
        icon = "✅" if r['improvement'] > 0 else ("🛡️" if r['plasticity'] < 0.1 else "❌")
        table_rows += f"| {i+1} | {r['name']} | {r['static_loss']:.4f} | {r['adaptive_loss']:.4f} | {r['plasticity']:.2f} | {r['improvement']:+.2f}% | {icon} |\n"

    report = f"""# MirrorMind V2 (The Beast) Report
**Date:** {timestamp}
**Mode:** Self-Regulating (Strong Framework)

## 1. Summary
* **Total Tasks:** {len(results)}
* **Average Improvement:** {avg_imp:.2f}%
* **Outcomes:** {wins} Wins | {skips} Shielded | {regressions} Regressions

## 2. Detailed Performance
| ID | Task | Static | Adaptive | Plasticity | Imp % | Status |
|:---|:---|:---|:---|:---|:---|:---|
{table_rows}
"""
    with open("PHASE5_V2_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("   ✅ Report generated: PHASE5_V2_REPORT.md")

# ==============================================================================
# 3. EXECUTION
# ==============================================================================
def run_strong_test():
    logger.info("🛡️ PHASE 5: THE BEAST EVALUATION (V2)")
    
    # Load Pretrained Weights
    PRETRAIN_PATH = "checkpoints/arc_pretrained_final.pt"
    
    # --- CHANGE: Instantiate V2 Model ---
    base_model = MirrorMindV2() 
    
    if os.path.exists(PRETRAIN_PATH):
        logger.info(f"   🧠 Loading Beast Brain: {PRETRAIN_PATH}")
        # Need strict=False if running on partial weights, 
        # but here we expect full match so strict=True is fine.
        base_model.load_state_dict(torch.load(PRETRAIN_PATH))
    else:
        logger.warning("   ⚠️ Pre-trained weights not found! Using Random Init.")

    results = []
    
    # CONFIGURATION: Strong Framework settings
    fw_config = AdaptiveFrameworkConfig(
        learning_rate=0.005, 
        enable_active_shield=True,  
        active_shield_threshold=0.085,
        active_shield_slope=20.0,
        compile_model=False,
        device='cpu' # Use CUDA if available
    )

    for i in range(100):
        task_data = load_eval_task("data/training")
        if not task_data: break
        sx, sy, qx, qy, name = task_data
        
        # 1. Baseline
        mm_model = copy.deepcopy(base_model)
        base_model.eval()
        with torch.no_grad():
            p_base = base_model(qx)
            l_base = nn.MSELoss()(p_base, qy).item()
            
        # 2. ADAPTATION (Active Shield + Tether)
        framework = AdaptiveFramework(mm_model, fw_config)
        
        # TIGHT TETHER for The Beast (2000.0)
        framework.ewc.lock_for_ttt(strength=2000.0)
        
        framework.model.train()
        last_plasticity = 0.0
        
        for _ in range(5):
            metrics = framework.train_step(sx, sy)
            last_plasticity = metrics.get('plasticity', 1.0)
            
        # 3. Evaluation
        framework.model.eval()
        with torch.no_grad():
            p_mm, _, _ = framework.forward(qx)
            l_mm = nn.MSELoss()(p_mm, qy).item()
            
        # 4. Scoring
        if l_base < 1e-9: l_base = 1e-9
        imp = ((l_base - l_mm) / l_base) * 100.0
        
        status = "🛡️ Shielded" if last_plasticity < 0.1 else "✅ Adapted"
        if l_mm > l_base and last_plasticity > 0.1: status = "❌ Regression"
            
        logger.info(f"   Task {i+1:02d}: Static={l_base:.4f} | Adap={l_mm:.4f} | Plas={last_plasticity:.2f} | {status}")
        
        results.append({
            'name': name,
            'static_loss': l_base,
            'adaptive_loss': l_mm,
            'plasticity': last_plasticity,
            'improvement': imp
        })

    generate_artifacts(results)

if __name__ == "__main__":
    run_strong_test()