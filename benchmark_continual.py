import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
import os
import argparse
import logging
import sys
import logging
import sys
import time
import traceback
from tqdm import tqdm # VISIBILITY: Progress bar

from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULTS_FILE = "benchmark_results.json"
CKPT_DIR = "checkpoints"

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("NEURIPS_BENCHMARK")

parser = argparse.ArgumentParser()
parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--force_restart', action='store_true')
args = parser.parse_args()

def get_cifar100_split():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))
    ])
    train = datasets.CIFAR100('./data', train=True, download=True, transform=transform)
    test = datasets.CIFAR100('./data', train=False, download=True, transform=transform)

    def idx(ds, cls): return [i for i,(_,y) in enumerate(ds) if y in cls]

    a_cls = list(range(50))
    b_cls = list(range(50,100))

    # SAFE BATCH SIZE FOR RTX 3050 (6GB)
    # 128 is too risky with ResNet + Optimizer + Backprop buffers
    bs = min(args.batch_size, 64) 
    
    nw = 4 if DEVICE=='cuda' else 0
    pm = DEVICE=='cuda'

    return {
        'A': {
            'train': DataLoader(Subset(train, idx(train,a_cls)), batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pm),
            'test':  DataLoader(Subset(test,  idx(test,a_cls)),  batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pm)
        },
        'B': {
            'train': DataLoader(Subset(train, idx(train,b_cls)), batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pm),
            'test':  DataLoader(Subset(test,  idx(test,b_cls)),  batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pm)
        }
    }

class ResNetLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AvgPool2d(8)
        )
        self.fc = nn.Linear(128,100)
    def forward(self,x):
        return self.fc(self.features(x).flatten(1))

def evaluate(model, loader):
    model.eval()
    c=t=0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            _,p = torch.max(model(x),1)
            t+=y.size(0); c+=(p==y).sum().item()
    return 100*c/t

def ckpt_path(seed, mode, task, epoch):
    d = f"{CKPT_DIR}/seed_{seed}/{mode}"
    os.makedirs(d, exist_ok=True)
    return f"{d}/task_{task}_epoch_{epoch}.pt"

def save_ckpt(path, model, task, epoch):
    torch.save({
        "model": model.state_dict(),
        "task": task,
        "epoch": epoch,
        "torch_rng": torch.get_rng_state(),
        "np_rng": np.random.get_state()
    }, path)

def load_latest_ckpt(seed, mode):
    d = f"{CKPT_DIR}/seed_{seed}/{mode}"
    if not os.path.exists(d): return None
    files = sorted(os.listdir(d))
    if not files: return None
    return os.path.join(d, files[-1])

def run(seed, data, mode):
    torch.manual_seed(seed); np.random.seed(seed)

    cfg = AdaptiveFrameworkConfig(device=DEVICE, learning_rate=0.001)
    if mode == 'base':
        cfg.memory_type='none'; cfg.enable_consciousness=False
    else:
        cfg.memory_type='hybrid'; cfg.enable_consciousness=True
        cfg.use_prioritized_replay=True; cfg.si_lambda=1.0

    model = AdaptiveFramework(ResNetLike(), cfg, device=DEVICE)

    start_task, start_epoch = 'A', 0
    ckpt = load_latest_ckpt(seed, mode)
    if ckpt:
        s = torch.load(ckpt, map_location=DEVICE , weights_only=False)
        model.load_state_dict(s["model"])
        start_task = s["task"]
        start_epoch = s["epoch"]+1
        torch.set_rng_state(s["torch_rng"])
        np.random.set_state(s["np_rng"])
        logger.info(f"Resumed {mode} seed {seed} from {start_task} epoch {start_epoch}")

    metrics_store = {
        "acc_task_a_phase1": 0.0,
        "acc_task_a_phase2": 0.0,
        "acc_task_b_phase2": 0.0
    }

    try:
        for task in ['A','B']:
            if task < start_task: continue
            current_start_epoch = start_epoch if task==start_task else 0
             
            for e in range(current_start_epoch, args.epochs):
                logger.info(f"[{mode.upper()}-{seed}] Task {task}: Epoch {e+1}/{args.epochs} started...")
                
                # VISIBILITY: Progress Bar
                pbar = tqdm(data[task]['train'], desc=f"Ep {e+1}/{args.epochs}", unit="batch")
                epoch_loss = 0.0
                
                for i, (x, y) in enumerate(pbar):
                    metrics = model.train_step(x.to(DEVICE), target_data=y.to(DEVICE))
                    
                    # Update pbar
                    loss_val = metrics.get('loss', 0.0) if isinstance(metrics, dict) else 0.0
                    epoch_loss += loss_val
                    pbar.set_postfix({"loss": f"{loss_val:.4f}"})
                
                avg_loss = epoch_loss / len(data[task]['train'])
                logger.info(f"[{mode.upper()}-{seed}] Task {task}: Epoch {e+1} Done. Avg Loss: {avg_loss:.4f}")
                
                # OPTIMIZATION: Aggressive VRAM cleanup for 3050
                if DEVICE=='cuda': torch.cuda.empty_cache()
                
                save_ckpt(ckpt_path(seed,mode,task,e), model, task, e)
                
            start_epoch = 0
            
            # PHASE 1 EVALUATION (After Task A)
            if task == 'A':
                logger.info("Evaluating Phase 1 (Post-Task A)...")
                metrics_store["acc_task_a_phase1"] = evaluate(model, data['A']['test'])
                # Also save consolidated checkpoint
                if mode=='airborne' and getattr(model, 'prioritized_buffer', None):
                    model.memory.consolidate(feedback_buffer=model.prioritized_buffer, current_step=1, mode='NORMAL')
                    if DEVICE=='cuda': torch.cuda.empty_cache()

    except Exception as e:
        logger.error("CRASH RECOVERED — checkpoint saved")
        logger.error(traceback.format_exc())
        raise e

    # PHASE 2 EVALUATION (After Task B)
    logger.info("Evaluating Phase 2 (Post-Task B)...")
    metrics_store["acc_task_a_phase2"] = evaluate(model, data['A']['test'])
    metrics_store["acc_task_b_phase2"] = evaluate(model, data['B']['test'])

    return metrics_store

if __name__ == "__main__":
    data = get_cifar100_split()
    results = json.load(open(RESULTS_FILE)) if os.path.exists(RESULTS_FILE) else {}
    for s in args.seeds:
        results.setdefault(str(s),{})
        for m in ['base','airborne']:
            if m in results[str(s)]: continue
            results[str(s)][m] = run(s,data,m)
            json.dump(results, open(RESULTS_FILE,'w'), indent=4)
