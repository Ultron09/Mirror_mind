"""
PROTOCOL PHASE 5 (STRONG): SELF-REGULATING ARC CHALLENGE
========================================================
Goal: Verify that the Framework can self-regulate plasticity without external scripts.
Changes:
1. REMOVED: The "if static_loss < 0.08" logic.
2. ADDED: Active Shield Configuration.
3. ADDED: Standard Safety Tethering.
"""

import torch
import torch.nn as nn
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
logger = logging.getLogger("Phase5_Strong" )

# ==============================================================================
# REPORTING UTILS
# ==============================================================================
def generate_artifacts(results):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Visualization
    try:
        static = [r['static_loss'] for r in results]
        adaptive = [r['adaptive_loss'] for r in results]
        
        plt.figure(figsize=(8, 8))
        plt.scatter(static, adaptive, c='blue', alpha=0.6, label='Tasks')
        max_val = max(max(static), max(adaptive))
        plt.plot([0, max_val], [0, max_val], 'r--', label='No Improvement')
        plt.fill_between([0, max_val], [0, max_val], 0, color='green', alpha=0.1, label='Improvement Zone')
        
        plt.title(f"MirrorMind Strong Framework (Gated)\n{timestamp}")
        plt.xlabel("Static Baseline Loss")
        plt.ylabel("Adaptive Loss (Shielded)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig("phase5_strong_performance.png", dpi=300)
        plt.close()
    except Exception:
        pass

    # 2. Markdown Report
    wins = sum(1 for r in results if r['improvement'] > 0.0001)
    skips = sum(1 for r in results if abs(r['improvement']) < 0.0001)
    regressions = sum(1 for r in results if r['improvement'] < -0.0001)
    avg_imp = np.mean([r['improvement'] for r in results])
    
    table_rows = ""
    for i, r in enumerate(results[:20]):
        icon = "✅" if r['improvement'] > 0 else ("🛡️" if abs(r['improvement']) < 1e-4 else "❌")
        table_rows += f"| {i+1} | {r['name']} | {r['static_loss']:.4f} | {r['adaptive_loss']:.4f} | {r['plasticity']:.2f} | {r['improvement']:+.2f}% | {icon} |\n"

    report = f"""# MirrorMind Strong Framework Report
**Date:** {timestamp}
**Mode:** Self-Regulating (Active Shield)

## 1. Summary
* **Total Tasks:** {len(results)}
* **Average Improvement:** {avg_imp:.2f}%
* **Outcomes:** {wins} Wins | {skips} Shielded (Skipped) | {regressions} Regressions

## 2. Detailed Performance
| ID | Task | Static | Adaptive | Plasticity | Imp % | Status |
|:---|:---|:---|:---|:---|:---|:---|
{table_rows}
"""
    with open("PHASE5_STRONG_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("   ✅ Report generated: PHASE5_STRONG_REPORT.md")

# ==============================================================================
# MODEL
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
        self.exit = nn.Conv2d(64, 10, 1)
        self.softmax = nn.Softmax(dim=1) 
    def forward(self, x):
        x = self.entry(x)
        x = self.body(x)
        x = self.exit(x)
        return self.softmax(x)

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

# ==============================================================================
# EXECUTION
# ==============================================================================
def run_strong_test():
    logger.info("🛡️ PHASE 5: STRONG FRAMEWORK TEST")
    
    # Load Pretrained Weights
    PRETRAIN_PATH = "checkpoints/arc_pretrained_final.pt"
    base_model = ResNetGridReasoner()
    if os.path.exists(PRETRAIN_PATH):
        logger.info(f"   🧠 Loaded Brain: {PRETRAIN_PATH}")
        base_model.load_state_dict(torch.load(PRETRAIN_PATH))
    else:
        logger.warning("   ⚠️ Using Random Weights")

    results = []
    
    # ---------------------------------------------------------
    # CONFIGURATION: THE INTELLIGENCE IS HERE
    # ---------------------------------------------------------
    fw_config = AdaptiveFrameworkConfig(
        learning_rate=0.005, 
        
        # 1. Active Shield: Internal Homeostasis
        enable_active_shield=True,  
        active_shield_threshold=0.085, # The "Boredom" Line
        active_shield_slope=20.0,
        
        compile_model=False,
        device='cpu'
    )
    # ---------------------------------------------------------

    for i in range(100):
        task_data = load_eval_task("data/training")
        if not task_data: break
        sx, sy, qx, qy, name = task_data
        
        # 1. Baseline (For Reporting Only - Framework doesn't see this!)
        mm_model = copy.deepcopy(base_model)
        base_model.eval()
        with torch.no_grad():
            p_base = base_model(qx)
            l_base = nn.MSELoss()(p_base, qy).item()
            
        # 2. ADAPTATION (Blind Execution)
        # We perform NO checks here. We just throw data at the framework.
        framework = AdaptiveFramework(mm_model, fw_config)
        
        # Standard Protocol: Always Lock Tether
        framework.ewc.lock_for_ttt(strength=500.0)
        
        framework.model.train()
        last_plasticity = 0.0
        
        for _ in range(5):
            # The Active Shield inside train_step decides if we learn
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