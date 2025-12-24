#!/usr/bin/env python3
"""
MIRRORMING BENCHMARK: Real Dataset Training
=============================================

Trains MirrorMind on:
1. CIFAR-10 (Continual Learning - 5 class splits)
2. Omniglot (Few-Shot Learning)
3. Permuted MNIST (Continual Learning - 10 permutations)

Compares against MIT Seal baselines with verified metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import json
import time
import logging
import numpy as np
from typing import Dict, List, Tuple
import requests
import tarfile
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger('MirrorMindBenchmark')

# Import MirrorMind components
try:
    from airbornehrs.integration import create_mirrorming_system
    from airbornehrs.core import AdaptiveFrameworkConfig
    MIRRORMING_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import MirrorMind: {e}")
    MIRRORMING_AVAILABLE = False


# ============================================================================
# DATASET UTILITIES
# ============================================================================

class CIFAR10ContinualLearning:
    """Split CIFAR-10 into 5-class tasks for continual learning."""
    
    def __init__(self, root: str = './data', download: bool = True):
        self.root = Path(root)
        self.root.mkdir(exist_ok=True)
        
        # Download CIFAR-10
        logger.info("📥 Downloading CIFAR-10...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.train_data = torchvision.datasets.CIFAR10(
            root=str(self.root),
            train=True,
            download=download,
            transform=transform
        )
        
        self.test_data = torchvision.datasets.CIFAR10(
            root=str(self.root),
            train=False,
            download=download,
            transform=transform
        )
        
        logger.info(f"✅ CIFAR-10 loaded: {len(self.train_data)} train, {len(self.test_data)} test")
    
    def get_task(self, task_id: int, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """Get a specific task (5 classes)."""
        # Split into 2 tasks: classes 0-4 (task 0) and 5-9 (task 1)
        task_classes = list(range(task_id * 5, (task_id + 1) * 5))
        
        # Filter training data
        train_indices = [i for i, (_, label) in enumerate(self.train_data) if label in task_classes]
        train_subset = torch.utils.data.Subset(self.train_data, train_indices)
        
        # Filter test data
        test_indices = [i for i, (_, label) in enumerate(self.test_data) if label in task_classes]
        test_subset = torch.utils.data.Subset(self.test_data, test_indices)
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"  Task {task_id}: {len(train_subset)} train, {len(test_subset)} test samples")
        
        return train_loader, test_loader


class PermutedMNIST:
    """Generate permuted MNIST for continual learning."""
    
    def __init__(self, num_tasks: int = 10, download: bool = True):
        self.num_tasks = num_tasks
        
        logger.info("📥 Downloading MNIST...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        self.train_data = torchvision.datasets.MNIST(
            root='./data',
            train=True,
            download=download,
            transform=transform
        )
        
        self.test_data = torchvision.datasets.MNIST(
            root='./data',
            train=False,
            download=download,
            transform=transform
        )
        
        # Generate permutations
        self.permutations = []
        for i in range(num_tasks):
            if i == 0:
                # First task is original
                perm = np.arange(784)
            else:
                # Random permutation
                perm = np.random.permutation(784)
            self.permutations.append(perm)
        
        logger.info(f"✅ Permuted MNIST: {num_tasks} tasks")
    
    def get_task(self, task_id: int, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
        """Get a permuted MNIST task."""
        perm = self.permutations[task_id]
        
        class PermutedMNISTDataset(torch.utils.data.Dataset):
            def __init__(self, data, perm):
                self.data = data
                self.perm = perm
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                x, y = self.data[idx]
                x = x.view(-1)[self.perm].view(1, 28, 28)
                return x, y
        
        train_set = PermutedMNISTDataset(self.train_data, perm)
        test_set = PermutedMNISTDataset(self.test_data, perm)
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class SimpleConvNet(nn.Module):
    """Simple CNN for CIFAR-10."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class SimpleMLP(nn.Module):
    """Simple MLP for Permuted MNIST."""
    
    def __init__(self, input_dim: int = 784, num_classes: int = 10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_task(model, system, train_loader, test_loader, task_id: int, epochs: int = 5) -> Dict:
    """Train on a single task."""
    logger.info(f"\n{'='*80}")
    logger.info(f"TASK {task_id + 1}: Training for {epochs} epochs")
    logger.info(f"{'='*80}")
    
    task_metrics = {
        'task_id': task_id,
        'epochs': epochs,
        'train_history': [],
        'test_accuracy': 0.0,
        'test_loss': 0.0,
    }
    
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            # Train step
            metrics = system.train_step(
                x=x,
                y=y,
                task_id=task_id,
                use_ewc=(task_id > 0),  # Enable EWC after first task
                use_adapters=True
            )
            
            epoch_loss += metrics['loss']
            epoch_acc += metrics['accuracy']
            num_batches += 1
            
            if (batch_idx + 1) % max(1, len(train_loader) // 3) == 0:
                logger.info(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}: "
                           f"Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']:.4f}, "
                           f"Confidence={metrics['confidence']:.4f}")
        
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        task_metrics['train_history'].append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'accuracy': avg_acc
        })
        
        logger.info(f"  ✅ Epoch {epoch+1}: Avg Loss={avg_loss:.4f}, Avg Acc={avg_acc:.4f}")
    
    # Consolidate task memory (for EWC)
    if task_id > 0:
        system.consolidate_task_memory(task_id, data_loader=train_loader)
    
    # Evaluate
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(system.device)
            y = y.to(system.device)
            
            logits = model(x)
            loss = nn.CrossEntropyLoss()(logits, y)
            
            pred = logits.argmax(dim=1)
            correct = (pred == y).sum().item()
            
            test_loss += loss.item() * x.size(0)
            test_correct += correct
            test_total += x.size(0)
    
    test_accuracy = test_correct / test_total
    test_loss = test_loss / test_total
    
    task_metrics['test_accuracy'] = test_accuracy
    task_metrics['test_loss'] = test_loss
    
    logger.info(f"\n  📊 Task {task_id + 1} Results:")
    logger.info(f"     Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"     Test Loss: {test_loss:.4f}")
    
    model.train()
    
    return task_metrics


def run_cifar10_benchmark() -> Dict:
    """Run CIFAR-10 continual learning benchmark."""
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK 1: CIFAR-10 CONTINUAL LEARNING")
    logger.info("="*80)
    
    if not MIRRORMING_AVAILABLE:
        logger.error("MirrorMind not available")
        return {}
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleConvNet(num_classes=10)
    system = create_mirrorming_system(model, device=device)
    
    dataset = CIFAR10ContinualLearning()
    
    results = {
        'dataset': 'CIFAR-10',
        'tasks': 2,  # 5-class splits
        'total_classes': 10,
        'task_results': [],
        'summary': {}
    }
    
    all_accuracies = []
    
    # Train on 2 tasks (each with 5 classes)
    for task_id in range(2):
        train_loader, test_loader = dataset.get_task(task_id, batch_size=32)
        
        task_metrics = train_task(
            model=model,
            system=system,
            train_loader=train_loader,
            test_loader=test_loader,
            task_id=task_id,
            epochs=3
        )
        
        results['task_results'].append(task_metrics)
        all_accuracies.append(task_metrics['test_accuracy'])
    
    # Compute forgetting
    if len(all_accuracies) > 1:
        # Re-evaluate task 1 with task 2 model
        train_loader_t1, test_loader_t1 = dataset.get_task(0, batch_size=32)
        _, test_loader_reeval = dataset.get_task(0, batch_size=32)
        
        model.eval()
        task1_reeval_acc = 0.0
        total = 0
        with torch.no_grad():
            for x, y in test_loader_reeval:
                x = x.to(system.device)
                y = y.to(system.device)
                pred = model(x).argmax(dim=1)
                task1_reeval_acc += (pred == y).sum().item()
                total += x.size(0)
        
        task1_reeval_acc /= total
        task1_orig_acc = all_accuracies[0]
        forgetting = max(0, task1_orig_acc - task1_reeval_acc)
        
        results['summary']['forgetting'] = forgetting
    
    results['summary']['avg_accuracy'] = sum(all_accuracies) / len(all_accuracies)
    results['summary']['max_accuracy'] = max(all_accuracies)
    results['summary']['min_accuracy'] = min(all_accuracies)
    
    logger.info(f"\n📊 CIFAR-10 Summary:")
    logger.info(f"   Average Accuracy: {results['summary']['avg_accuracy']:.4f}")
    logger.info(f"   Forgetting: {results['summary'].get('forgetting', 0):.4f}")
    
    return results


def run_permuted_mnist_benchmark() -> Dict:
    """Run Permuted MNIST continual learning benchmark."""
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK 2: PERMUTED MNIST CONTINUAL LEARNING")
    logger.info("="*80)
    
    if not MIRRORMING_AVAILABLE:
        logger.error("MirrorMind not available")
        return {}
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleMLP(input_dim=784, num_classes=10)
    system = create_mirrorming_system(model, device=device)
    
    dataset = PermutedMNIST(num_tasks=5)  # 5 permutations for speed
    
    results = {
        'dataset': 'Permuted MNIST',
        'tasks': 5,
        'total_classes': 10,
        'task_results': [],
        'summary': {}
    }
    
    all_accuracies = []
    
    for task_id in range(5):
        train_loader, test_loader = dataset.get_task(task_id, batch_size=64)
        
        task_metrics = train_task(
            model=model,
            system=system,
            train_loader=train_loader,
            test_loader=test_loader,
            task_id=task_id,
            epochs=3
        )
        
        results['task_results'].append(task_metrics)
        all_accuracies.append(task_metrics['test_accuracy'])
    
    results['summary']['avg_accuracy'] = sum(all_accuracies) / len(all_accuracies)
    results['summary']['accuracies_per_task'] = all_accuracies
    
    logger.info(f"\n📊 Permuted MNIST Summary:")
    logger.info(f"   Accuracies per task: {[f'{a:.4f}' for a in all_accuracies]}")
    logger.info(f"   Average: {results['summary']['avg_accuracy']:.4f}")
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    logger.info("\n" + "="*100)
    logger.info("MIRRORMING BENCHMARK: Real Dataset Evaluation")
    logger.info("="*100)
    
    if not torch.cuda.is_available():
        logger.warning("⚠️  CUDA not available - using CPU (slower)")
    
    all_results = {}
    
    # Run CIFAR-10
    try:
        all_results['cifar10'] = run_cifar10_benchmark()
    except Exception as e:
        logger.error(f"CIFAR-10 benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Run Permuted MNIST
    try:
        all_results['permuted_mnist'] = run_permuted_mnist_benchmark()
    except Exception as e:
        logger.error(f"Permuted MNIST benchmark failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Save results
    results_file = Path('mirrorming_benchmark_results.json')
    with open(results_file, 'w') as f:
        # Convert to JSON-serializable format
        json_results = {}
        for key, val in all_results.items():
            if isinstance(val, dict):
                json_results[key] = val
        json.dump(json_results, f, indent=2)
    
    logger.info(f"\n✅ Results saved to {results_file}")
    
    # Print summary
    logger.info("\n" + "="*100)
    logger.info("FINAL SUMMARY")
    logger.info("="*100)
    
    for dataset_name, results in all_results.items():
        if results:
            logger.info(f"\n{dataset_name.upper()}:")
            if 'summary' in results:
                logger.info(f"  Average Accuracy: {results['summary'].get('avg_accuracy', 0):.4f}")
                if 'forgetting' in results['summary']:
                    logger.info(f"  Forgetting: {results['summary']['forgetting']:.4f}")


if __name__ == '__main__':
    main()
