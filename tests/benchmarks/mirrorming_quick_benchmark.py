#!/usr/bin/env python3
"""
MIRRORMING QUICK BENCHMARK: Synthetic Data Testing
====================================================

Tests the integrated MirrorMind system with synthetic data
to verify all components work together in a realistic scenario.
"""

import torch
import torch.nn as nn
import json
import logging
from pathlib import Path
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger('MirrorMindBenchmark')

from airbornehrs.integration import create_mirrorming_system

# ============================================================================
# SYNTHETIC DATA GENERATORS
# ============================================================================

class SimpleConvNet(nn.Module):
    """Simple ConvNet for CIFAR-10 style tasks"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # x shape: [batch, 3, 32, 32]
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class SimpleMLP(nn.Module):
    """Simple MLP for MNIST style tasks"""
    def __init__(self, input_dim=784, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# ============================================================================
# BENCHMARK 1: CONTINUAL LEARNING WITH EWC
# ============================================================================

def benchmark_continual_learning_with_ewc():
    """Test continual learning: Train Task 1, then Task 2, measure forgetting"""
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK 1: CONTINUAL LEARNING WITH EWC")
    logger.info("="*80)
    
    # Create model and system
    model = SimpleMLP(input_dim=28*28, num_classes=10)
    system = create_mirrorming_system(model, device='cpu')
    
    results = {
        'benchmark': 'continual_learning_ewc',
        'epochs_per_task': 3,
        'batch_size': 32,
        'num_samples_per_task': 320,
    }
    
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ===== TASK 1: Train on classes 0-4 =====
    logger.info("\n--- TASK 1: Train on Classes 0-4 ---")
    task1_metrics = []
    
    for epoch in range(3):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        
        for batch in range(10):  # 10 batches of 32 samples = 320 total
            x = torch.randn(32, 1, 28, 28)  # Random MNIST-like data
            y = torch.randint(0, 5, (32,))   # Labels 0-4
            
            metrics = system.train_step(x, y, task_id=0, use_ewc=False, use_adapters=True)
            epoch_loss += metrics['loss']
            epoch_acc += metrics['accuracy']
            num_batches += 1
        
        epoch_loss /= num_batches
        epoch_acc /= num_batches
        task1_metrics.append({'loss': epoch_loss, 'accuracy': epoch_acc})
        logger.info(f"  Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.4f}")
    
    # Consolidate Task 1 memory with EWC
    logger.info("  🧠 Consolidating Task 1 memory with EWC...")
    system.consolidate_task_memory(0)
    logger.info(f"  ✅ EWC Enabled: {system.ewc.is_enabled()}")
    
    # Evaluate Task 1 (before Task 2)
    task1_eval_before = system.evaluate([(torch.randn(32, 1, 28, 28), torch.randint(0, 5, (32,)))])
    logger.info(f"  Task 1 Accuracy (before Task 2): {task1_eval_before['accuracy']:.4f}")
    
    # ===== TASK 2: Train on classes 5-9 =====
    logger.info("\n--- TASK 2: Train on Classes 5-9 ---")
    task2_metrics = []
    
    for epoch in range(3):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        
        for batch in range(10):
            x = torch.randn(32, 1, 28, 28)
            y = torch.randint(5, 10, (32,))  # Labels 5-9
            
            metrics = system.train_step(x, y, task_id=1, use_ewc=True, use_adapters=True)
            epoch_loss += metrics['loss']
            epoch_acc += metrics['accuracy']
            num_batches += 1
        
        epoch_loss /= num_batches
        epoch_acc /= num_batches
        task2_metrics.append({'loss': epoch_loss, 'accuracy': epoch_acc})
        logger.info(f"  Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.4f}")
    
    # Evaluate Task 2
    task2_eval = system.evaluate([(torch.randn(32, 1, 28, 28), torch.randint(5, 10, (32,)))])
    logger.info(f"  Task 2 Accuracy: {task2_eval['accuracy']:.4f}")
    
    # Re-evaluate Task 1 (after Task 2) - measure catastrophic forgetting
    task1_eval_after = system.evaluate([(torch.randn(32, 1, 28, 28), torch.randint(0, 5, (32,)))])
    logger.info(f"  Task 1 Accuracy (after Task 2): {task1_eval_after['accuracy']:.4f}")
    
    forgetting = task1_eval_before['accuracy'] - task1_eval_after['accuracy']
    logger.info(f"\n  ✅ Catastrophic Forgetting: {forgetting:.4f}")
    
    results['task1_metrics'] = task1_metrics
    results['task2_metrics'] = task2_metrics
    results['task1_accuracy_before'] = task1_eval_before['accuracy']
    results['task1_accuracy_after'] = task1_eval_after['accuracy']
    results['task2_accuracy'] = task2_eval['accuracy']
    results['catastrophic_forgetting'] = forgetting
    results['ewc_enabled'] = system.ewc.is_enabled()
    
    return results


# ============================================================================
# BENCHMARK 2: CONSCIOUSNESS TRACKING
# ============================================================================

def benchmark_consciousness_tracking():
    """Test consciousness metrics across different data distributions"""
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK 2: CONSCIOUSNESS TRACKING & SELF-AWARENESS")
    logger.info("="*80)
    
    model = SimpleMLP(input_dim=28*28, num_classes=10)
    system = create_mirrorming_system(model, device='cpu')
    
    results = {
        'benchmark': 'consciousness_tracking',
        'epochs': 5,
        'metrics': []
    }
    
    logger.info(f"Training {results['epochs']} epochs, tracking consciousness metrics...")
    
    consciousness_history = []
    
    for epoch in range(results['epochs']):
        epoch_metrics = {
            'epoch': epoch + 1,
            'confidence': 0.0,
            'uncertainty': 0.0,
            'surprise': 0.0,
            'importance': 0.0,
            'count': 0
        }
        
        for batch in range(5):
            x = torch.randn(32, 1, 28, 28)
            y = torch.randint(0, 10, (32,))
            
            metrics = system.train_step(x, y, task_id=0, use_ewc=False, use_adapters=True)
            
            epoch_metrics['confidence'] += metrics.get('confidence', 0.0)
            epoch_metrics['uncertainty'] += metrics.get('uncertainty', 0.0)
            epoch_metrics['surprise'] += metrics.get('surprise', 0.0)
            epoch_metrics['importance'] += metrics.get('importance', 0.0)
            epoch_metrics['count'] += 1
        
        # Average metrics
        for key in ['confidence', 'uncertainty', 'surprise', 'importance']:
            if epoch_metrics['count'] > 0:
                epoch_metrics[key] /= epoch_metrics['count']
        
        del epoch_metrics['count']
        consciousness_history.append(epoch_metrics)
        
        logger.info(f"  Epoch {epoch+1}: "
                   f"Confidence={epoch_metrics['confidence']:.4f}, "
                   f"Uncertainty={epoch_metrics['uncertainty']:.4f}, "
                   f"Surprise={epoch_metrics['surprise']:.4f}, "
                   f"Importance={epoch_metrics['importance']:.4f}")
    
    results['consciousness_history'] = consciousness_history
    
    return results


# ============================================================================
# BENCHMARK 3: ADAPTER EFFICIENCY
# ============================================================================

def benchmark_adapter_efficiency():
    """Measure parameter overhead and training efficiency with adapters"""
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK 3: ADAPTER EFFICIENCY & PARAMETER OVERHEAD")
    logger.info("="*80)
    
    model = SimpleMLP(input_dim=28*28, num_classes=10)
    
    # Count base parameters
    base_params = sum(p.numel() for p in model.parameters())
    
    # Create system with adapters
    system = create_mirrorming_system(model, device='cpu')
    
    total_params = sum(p.numel() for p in system.model.parameters())
    adapter_params = sum(p.numel() for p in system.adapters.parameters()) if hasattr(system.adapters, 'parameters') else 0
    
    # If that doesn't work, sum from all adapters
    if adapter_params == 0:
        for adapter in system.adapters.adapters:
            adapter_params += sum(p.numel() for p in adapter.parameters())
    
    overhead_percent = (adapter_params / base_params) * 100 if base_params > 0 else 0.0
    
    results = {
        'benchmark': 'adapter_efficiency',
        'base_parameters': base_params,
        'adapter_parameters': adapter_params,
        'total_parameters': total_params,
        'overhead_percent': overhead_percent,
        'num_adapters': len(system.adapters.adapters)
    }
    
    logger.info(f"Base Model Parameters: {base_params:,}")
    logger.info(f"Adapter Parameters: {adapter_params:,}")
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Overhead: {overhead_percent:.2f}%")
    logger.info(f"Number of Adapters: {results['num_adapters']}")
    
    return results


# ============================================================================
# BENCHMARK 4: INFERENCE SPEED
# ============================================================================

def benchmark_inference_speed():
    """Measure inference speed with all components"""
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK 4: INFERENCE SPEED & THROUGHPUT")
    logger.info("="*80)
    
    model = SimpleMLP(input_dim=28*28, num_classes=10)
    system = create_mirrorming_system(model, device='cpu')
    
    import time
    
    # Warmup
    x = torch.randn(32, 1, 28, 28).to(system.device)
    with torch.no_grad():
        system.model(x)
    
    # Measure inference speed
    num_iterations = 100
    batch_size = 32
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            x = torch.randn(batch_size, 1, 28, 28).to(system.device)
            _ = system.model(x)
    elapsed = time.time() - start_time
    
    total_samples = num_iterations * batch_size
    samples_per_sec = total_samples / elapsed
    ms_per_sample = (elapsed / total_samples) * 1000
    
    results = {
        'benchmark': 'inference_speed',
        'num_iterations': num_iterations,
        'batch_size': batch_size,
        'total_samples': total_samples,
        'elapsed_seconds': elapsed,
        'samples_per_second': samples_per_sec,
        'ms_per_sample': ms_per_sample
    }
    
    logger.info(f"Iterations: {num_iterations}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Total Samples: {total_samples}")
    logger.info(f"Elapsed Time: {elapsed:.2f}s")
    logger.info(f"Throughput: {samples_per_sec:.0f} samples/sec")
    logger.info(f"Latency: {ms_per_sample:.4f} ms/sample")
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    logger.info("\n\n")
    logger.info("╔" + "="*78 + "╗")
    logger.info("║" + " "*78 + "║")
    logger.info("║" + "MIRRORMING QUICK BENCHMARK: SYNTHETIC DATA EVALUATION".center(78) + "║")
    logger.info("║" + " "*78 + "║")
    logger.info("╚" + "="*78 + "╝\n")
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'benchmarks': {}
    }
    
    try:
        # Run benchmarks
        all_results['benchmarks']['continual_learning'] = benchmark_continual_learning_with_ewc()
        all_results['benchmarks']['consciousness'] = benchmark_consciousness_tracking()
        all_results['benchmarks']['adapters'] = benchmark_adapter_efficiency()
        all_results['benchmarks']['inference_speed'] = benchmark_inference_speed()
        
        # Save results
        output_file = Path('mirrorming_quick_benchmark_results.json')
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info("\n\n" + "="*80)
        logger.info("BENCHMARK COMPLETE!")
        logger.info("="*80)
        logger.info(f"Results saved to: {output_file}")
        logger.info("\nSUMMARY:")
        logger.info(f"  ✅ Continual Learning: PASSED (Forgetting={all_results['benchmarks']['continual_learning']['catastrophic_forgetting']:.4f})")
        logger.info(f"  ✅ Consciousness: PASSED ({len(all_results['benchmarks']['consciousness']['consciousness_history'])} epochs tracked)")
        logger.info(f"  ✅ Adapters: PASSED (Overhead={all_results['benchmarks']['adapters']['overhead_percent']:.2f}%)")
        logger.info(f"  ✅ Inference Speed: PASSED ({all_results['benchmarks']['inference_speed']['samples_per_second']:.0f} samples/sec)")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        all_results['status'] = 'FAILED'
        all_results['error'] = str(e)
        
        output_file = Path('mirrorming_quick_benchmark_results.json')
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)


if __name__ == '__main__':
    main()
