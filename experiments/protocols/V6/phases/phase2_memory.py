
"""
PROTOCOL V6 - PHASE 2: VERIFICATION (MEMORY)
============================================
Goal: Verify resistance to catastrophic forgetting.
Dataset: Sequential Split-CIFAR10 (5 Tasks: [0,1], [2,3], [4,5], [6,7], [8,9]).
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import logging

# Path Setup
# Path Setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from airbornehrs.core import AdaptiveFramework, AdaptiveFrameworkConfig

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger("Phase2")

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10) # 10 classes total

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_split_cifar10(task_id, train=True):
    """Returns a subset of CIFAR10 for the given task_id (0-4)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    
    classes = [task_id * 2, task_id * 2 + 1]
    indices = [i for i, target in enumerate(dataset.targets) if target in classes]
    return Subset(dataset, indices)

def run_phase2():
    logger.info("🧠 PHASE 2: SPLIT-CIFAR10 (MEMORY)")
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Setup
    model = ConvNet().to(DEVICE)
    config = AdaptiveFrameworkConfig(
        learning_rate=0.001,
        enable_consciousness=True,
        device=DEVICE,
        memory_type='si', # Synaptic Intelligence
        dream_interval=50, # Frequent replay
        consciousness_buffer_size=10000
    )
    framework = AdaptiveFramework(model, config)
    
    # Metrics
    accuracies = np.zeros((5, 5)) # rows=tasks trained, cols=tasks evaluated
    
    # Train sequentially on 5 tasks
    for task_id in range(5):
        logger.info(f"\n📚 Training Task {task_id} (Classes {task_id*2}-{task_id*2+1})...")
        
        train_loader = DataLoader(get_split_cifar10(task_id, train=True), batch_size=64, shuffle=True)
        
        framework.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            framework.train_step(data, target_data=target)
            if batch_idx >= 100: break # Short training
            
        # Evaluate on ALL tasks seen so far
        logger.info("   Evaluating on all tasks...")
        for eval_task in range(5):
            test_loader = DataLoader(get_split_cifar10(eval_task, train=False), batch_size=1000, shuffle=False)
            
            framework.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(DEVICE), target.to(DEVICE)
                    output, _, _ = framework(data)
                    if isinstance(output, tuple): output = output[0]
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
            
            acc = correct / total
            accuracies[task_id, eval_task] = acc
            logger.info(f"      Task {eval_task} Acc: {acc:.2%}")
            
    # Visualization
    plt.figure(figsize=(8, 6))
    plt.imshow(accuracies, interpolation='nearest', cmap='viridis')
    plt.title("Phase 2: Memory Retention Matrix")
    plt.xlabel("Evaluated Task")
    plt.ylabel("Training Task")
    plt.colorbar(label="Accuracy")
    plt.xticks(range(5), [f"T{i}" for i in range(5)])
    plt.yticks(range(5), [f"T{i}" for i in range(5)])
    plt.savefig("phase2_results.png")
    logger.info("   ✅ Plot saved: phase2_results.png")
    
    # Analysis
    # Forgetting = Max Acc - Final Acc for Task 0
    max_acc_t0 = np.max(accuracies[:, 0])
    final_acc_t0 = accuracies[4, 0]
    forgetting = max_acc_t0 - final_acc_t0
    
    logger.info(f"\nTask 0 Max Acc: {max_acc_t0:.2%}")
    logger.info(f"Task 0 Final Acc: {final_acc_t0:.2%}")
    logger.info(f"Forgetting: {forgetting:.2%}")
    
    if forgetting < 0.20:
        logger.info("✅ PHASE 2 PASSED: Low catastrophic forgetting.")
    else:
        logger.warning("⚠️ PHASE 2 WARNING: Significant forgetting detected.")

if __name__ == "__main__":
    run_phase2()
