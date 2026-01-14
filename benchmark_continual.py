import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
import os
import argparse
import logging
import time
from datetime import datetime

# Import AirborneHRS
from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig

# Configure Logging
import sys
# Force UTF-8 for stdout/stderr
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

# ==================== ARGUMENTS ====================
parser = argparse.ArgumentParser(description="NeurIPS Continual Learning Benchmark")
parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46], help='Random seeds (Recommend 5 for paper)')
parser.add_argument('--epochs', type=int, default=5, help='Epochs per task')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--force_restart', action='store_true', help='Overwrite existing results')
args = parser.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULTS_FILE = "benchmark_results.json"

# ==================== DATASET SETUP ====================
def get_cifar100_split():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    logger.info("Loading CIFAR-100...")
    train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    
    def get_indices(dataset, classes):
        return [i for i, (_, label) in enumerate(dataset) if label in classes]

    task_a_classes = list(range(0, 50))
    task_b_classes = list(range(50, 100))
    
    # OPTIMIZATION: Faster Data Loading
    num_workers = 4 if torch.cuda.is_available() else 0
    pin_memory = True if torch.cuda.is_available() else False
    
    # OPTIMIZATION: Bigger Batch Size (512)
    active_batch_size = max(args.batch_size, 512)

    loaders = {
        'A': {
            'train': DataLoader(Subset(train_set, get_indices(train_set, task_a_classes)), 
                              batch_size=active_batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=2 if num_workers>0 else None),
            'test': DataLoader(Subset(test_set, get_indices(test_set, task_a_classes)), 
                             batch_size=active_batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
        },
        'B': {
            'train': DataLoader(Subset(train_set, get_indices(train_set, task_b_classes)), 
                              batch_size=active_batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=2 if num_workers>0 else None),
            'test': DataLoader(Subset(test_set, get_indices(test_set, task_b_classes)), 
                             batch_size=active_batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
        }
    }
    return loaders

# ==================== MODEL ====================
class ResNetLike(nn.Module):
    """A slightly deeper model for CIFAR-100 to show real capacity."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AvgPool2d(8) # Global Average Pooling
        )
        self.classifier = nn.Linear(128, 100)
        
    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

# ==================== EVALUATION ====================
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            if isinstance(outputs, tuple): outputs = outputs[0]
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return 100 * correct / total

# ==================== EXPERIMENT RUNNER ====================
def run_experiment_seed(seed, dataset, mode):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # OPTIMIZATION: Enable CuDNN Benchmarking
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        
    logger.info(f"--- STARTING: {mode.upper()} | SEED: {seed} ---")
    
    base_model = ResNetLike()
    
    # Configuration
    config = AdaptiveFrameworkConfig(
        device=DEVICE,
        learning_rate=0.001,
        evaluation_frequency=1000 # Reduce log spam
    )
    
    if mode == 'naive':
        config.memory_type = 'none' 
        config.enable_consciousness = False
    elif mode == 'ewc':
        config.memory_type = 'ewc'
        config.enable_consciousness = False
        config.ewc_lambda = 2000.0 # Standard strong EWC
    elif mode == 'airborne':
        config.memory_type = 'hybrid'
        config.enable_consciousness = True
        config.si_lambda = 1.0
        config.use_prioritized_replay = True # Full features
    
    framework = AdaptiveFramework(base_model, config, device=DEVICE)
    
    # TASK A
    logger.info(f"[{mode.upper()}-{seed}] Training Task A ({args.epochs} epo)...")
    t0 = time.time()
    for epoch in range(args.epochs):
        for x, y in dataset['A']['train']:
            framework.train_step(x.to(DEVICE), target_data=y.to(DEVICE))
    logger.info(f"[{mode.upper()}-{seed}] Task A done in {time.time()-t0:.1f}s")
    
    acc_a_p1 = evaluate(framework, dataset['A']['test'])
    
    # CONSOLIDATION
    if mode != 'naive':
        logger.info(f"[{mode.upper()}-{seed}] Consolidating Memory...")
        # FIX: Pass the buffer explicitly for Fisher Matrix calculation
        framework.memory.consolidate(
            feedback_buffer=framework.prioritized_buffer,
            current_step=1, 
            mode='NORMAL'
        )

    # TASK B
    logger.info(f"[{mode.upper()}-{seed}] Training Task B ({args.epochs} epo)...")
    for epoch in range(args.epochs):
        for x, y in dataset['B']['train']:
            framework.train_step(x.to(DEVICE), target_data=y.to(DEVICE))
            
    # FINAL EVAL
    acc_a_p2 = evaluate(framework, dataset['A']['test'])
    acc_b_p2 = evaluate(framework, dataset['B']['test'])
    bwt = acc_a_p2 - acc_a_p1
    
    logger.info(f"[{mode.upper()}-{seed}] RESULT: A1={acc_a_p1:.2f} | A2={acc_a_p2:.2f} | B2={acc_b_p2:.2f} | BWT={bwt:.2f}")
    
    return {
        "acc_task_a_phase1": acc_a_p1,
        "acc_task_a_phase2": acc_a_p2,
        "acc_task_b_phase2": acc_b_p2,
        "bwt": bwt
    }

# ==================== MAIN ====================
if __name__ == "__main__":
    if args.force_restart and os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)
        
    # Load existing
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            results = json.load(f)
        logger.info(f"Resumed from {RESULTS_FILE}")
    else:
        results = {}
        
    dataset = get_cifar100_split()
    modes = ['naive', 'ewc', 'airborne']
    
    try:
        for seed in args.seeds:
            seed_str = str(seed)
            if seed_str not in results:
                results[seed_str] = {}
                
            for mode in modes:
                if mode in results[seed_str]:
                    logger.info(f"Skipping {mode}-{seed} (Already done)")
                    continue
                    
                # Run
                metrics = run_experiment_seed(seed, dataset, mode)
                
                # Save immediately
                results[seed_str][mode] = metrics
                with open(RESULTS_FILE, 'w') as f:
                    json.dump(results, f, indent=4)
        
        logger.info("All experiments completed successfully.")
        
    except KeyboardInterrupt:
        logger.warning("Benchmark interrupted by user. Progress saved.")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
