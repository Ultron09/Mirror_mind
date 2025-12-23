"""
Protocol v2: Multi-Modality Tests
=================================
Tests consciousness and memory across different data types:
- Vision (image-like 2D tensors)
- Text (sequence embeddings)
- Mixed modalities (concatenated features)
- Different input/output dimensionalities
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict
import logging

# Disable logging to avoid Windows encoding issues
logging.disable(logging.CRITICAL)


class MultiModalityTester:
    """Test framework with different data modalities."""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'modalities_tested': {},
            'tests_passed': 0,
            'tests_failed': 0
        }
    
    def test_vision_modality(self):
        """Test 1: Vision data (simulated images)."""
        print("\n[TEST 1] Vision Modality (Images)")
        try:
            from airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig
            
            # Simulate image data: 28x28 flattened = 784
            vision_dim = 784  # MNIST-like
            output_dim = 10   # 10 classes
            
            config = AdaptiveFrameworkConfig(
                enable_consciousness=True,
                memory_type='hybrid',
                device=self.device
            )
            
            model = nn.Sequential(
                nn.Linear(vision_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim)
            ).to(self.device)
            
            framework = AdaptiveFramework(model, config)
            
            # Train on vision data
            losses = []
            torch.manual_seed(42)
            for step in range(50):
                # Simulate image batch: [batch_size, 784]
                x = torch.rand(16, vision_dim, device=self.device)
                y = torch.randn(16, output_dim, device=self.device)
                
                metrics = framework.train_step(x, y, enable_dream=False)
                losses.append(metrics.get('loss', 0.0))
            
            avg_loss = np.mean(losses)
            improvement = (losses[0] - losses[-1]) / losses[0] if losses[0] > 0 else 0
            
            print(f"    [OK] Vision: 50 steps | Loss: {avg_loss:.4f} | Improvement: {improvement*100:.1f}%")
            
            self.results['modalities_tested']['vision'] = {
                'dimension': vision_dim,
                'output_dim': output_dim,
                'steps': 50,
                'final_loss': float(avg_loss),
                'improvement': float(improvement)
            }
            self.results['tests_passed'] += 1
            return True
        except Exception as e:
            print(f"    [FAIL] {e}")
            self.results['tests_failed'] += 1
            return False
    
    def test_text_modality(self):
        """Test 2: Text data (word embeddings)."""
        print("\n[TEST 2] Text Modality (Embeddings)")
        try:
            from airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig
            
            # Simulate text embeddings: 768-dim (BERT-like)
            embedding_dim = 768
            output_dim = 2  # Binary classification (e.g., sentiment)
            
            config = AdaptiveFrameworkConfig(
                enable_consciousness=True,
                memory_type='hybrid',
                device=self.device
            )
            
            model = nn.Sequential(
                nn.Linear(embedding_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim)
            ).to(self.device)
            
            framework = AdaptiveFramework(model, config)
            
            # Train on text data
            losses = []
            torch.manual_seed(42)
            for step in range(50):
                # Simulate text embeddings: [batch_size, 768]
                x = torch.randn(16, embedding_dim, device=self.device)
                y = torch.randn(16, output_dim, device=self.device)
                
                metrics = framework.train_step(x, y, enable_dream=False)
                losses.append(metrics.get('loss', 0.0))
            
            avg_loss = np.mean(losses)
            improvement = (losses[0] - losses[-1]) / losses[0] if losses[0] > 0 else 0
            
            print(f"    [OK] Text: 50 steps | Loss: {avg_loss:.4f} | Improvement: {improvement*100:.1f}%")
            
            self.results['modalities_tested']['text'] = {
                'dimension': embedding_dim,
                'output_dim': output_dim,
                'steps': 50,
                'final_loss': float(avg_loss),
                'improvement': float(improvement)
            }
            self.results['tests_passed'] += 1
            return True
        except Exception as e:
            print(f"    [FAIL] {e}")
            self.results['tests_failed'] += 1
            return False
    
    def test_mixed_modality(self):
        """Test 3: Mixed modality (vision + text combined)."""
        print("\n[TEST 3] Mixed Modality (Vision + Text)")
        try:
            from airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig
            
            # Combine vision (784) + text (256) features
            vision_dim = 784
            text_dim = 256
            combined_dim = vision_dim + text_dim  # 1040
            output_dim = 10
            
            config = AdaptiveFrameworkConfig(
                enable_consciousness=True,
                memory_type='hybrid',
                device=self.device
            )
            
            model = nn.Sequential(
                nn.Linear(combined_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim)
            ).to(self.device)
            
            framework = AdaptiveFramework(model, config)
            
            # Train on mixed data
            losses = []
            torch.manual_seed(42)
            for step in range(50):
                # Vision part
                x_vision = torch.rand(16, vision_dim, device=self.device)
                # Text part
                x_text = torch.randn(16, text_dim, device=self.device)
                # Concatenate
                x = torch.cat([x_vision, x_text], dim=1)
                y = torch.randn(16, output_dim, device=self.device)
                
                metrics = framework.train_step(x, y, enable_dream=False)
                losses.append(metrics.get('loss', 0.0))
            
            avg_loss = np.mean(losses)
            improvement = (losses[0] - losses[-1]) / losses[0] if losses[0] > 0 else 0
            
            print(f"    [OK] Mixed: 50 steps | Loss: {avg_loss:.4f} | Improvement: {improvement*100:.1f}%")
            
            self.results['modalities_tested']['mixed'] = {
                'vision_dim': vision_dim,
                'text_dim': text_dim,
                'combined_dim': combined_dim,
                'output_dim': output_dim,
                'steps': 50,
                'final_loss': float(avg_loss),
                'improvement': float(improvement)
            }
            self.results['tests_passed'] += 1
            return True
        except Exception as e:
            print(f"    [FAIL] {e}")
            self.results['tests_failed'] += 1
            return False
    
    def test_high_dimensional_input(self):
        """Test 4: Very high-dimensional input."""
        print("\n[TEST 4] High-Dimensional Input")
        try:
            from airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig
            
            # Test with 4096-dimensional input
            high_dim = 4096
            output_dim = 512
            
            config = AdaptiveFrameworkConfig(
                enable_consciousness=True,
                memory_type='hybrid',
                device=self.device
            )
            
            model = nn.Sequential(
                nn.Linear(high_dim, 1024),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim)
            ).to(self.device)
            
            framework = AdaptiveFramework(model, config)
            
            # Train
            losses = []
            torch.manual_seed(42)
            for step in range(30):
                x = torch.randn(16, high_dim, device=self.device)
                y = torch.randn(16, output_dim, device=self.device)
                
                metrics = framework.train_step(x, y, enable_dream=False)
                losses.append(metrics.get('loss', 0.0))
            
            avg_loss = np.mean(losses)
            improvement = (losses[0] - losses[-1]) / losses[0] if losses[0] > 0 else 0
            
            print(f"    [OK] High-dim (4096): 30 steps | Loss: {avg_loss:.4f} | Improvement: {improvement*100:.1f}%")
            
            self.results['modalities_tested']['high_dim'] = {
                'input_dim': high_dim,
                'output_dim': output_dim,
                'steps': 30,
                'final_loss': float(avg_loss),
                'improvement': float(improvement)
            }
            self.results['tests_passed'] += 1
            return True
        except Exception as e:
            print(f"    [FAIL] {e}")
            self.results['tests_failed'] += 1
            return False
    
    def test_time_series_modality(self):
        """Test 5: Time series data (reshaped to vectors)."""
        print("\n[TEST 5] Time Series Modality")
        try:
            from airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig
            
            # Time series: sequence_length * features flattened
            sequence_length = 50
            num_features = 8
            ts_dim = sequence_length * num_features  # 400
            output_dim = 1
            
            config = AdaptiveFrameworkConfig(
                enable_consciousness=True,
                memory_type='hybrid',
                device=self.device
            )
            
            model = nn.Sequential(
                nn.Linear(ts_dim, 200),
                nn.ReLU(),
                nn.Linear(200, 100),
                nn.ReLU(),
                nn.Linear(100, output_dim)
            ).to(self.device)
            
            framework = AdaptiveFramework(model, config)
            
            # Train on time series data
            losses = []
            torch.manual_seed(42)
            for step in range(50):
                # Generate time series: [batch, seq_len, features]
                x_ts = torch.randn(16, sequence_length, num_features, device=self.device)
                # Flatten for MLP
                x = x_ts.reshape(16, ts_dim)
                y = torch.randn(16, output_dim, device=self.device)
                
                metrics = framework.train_step(x, y, enable_dream=False)
                losses.append(metrics.get('loss', 0.0))
            
            avg_loss = np.mean(losses)
            improvement = (losses[0] - losses[-1]) / losses[0] if losses[0] > 0 else 0
            
            print(f"    [OK] Time Series: 50 steps | Loss: {avg_loss:.4f} | Improvement: {improvement*100:.1f}%")
            
            self.results['modalities_tested']['time_series'] = {
                'sequence_length': sequence_length,
                'num_features': num_features,
                'flattened_dim': ts_dim,
                'output_dim': output_dim,
                'steps': 50,
                'final_loss': float(avg_loss),
                'improvement': float(improvement)
            }
            self.results['tests_passed'] += 1
            return True
        except Exception as e:
            print(f"    [FAIL] {e}")
            self.results['tests_failed'] += 1
            return False
    
    def run_all(self):
        """Run all multi-modality tests."""
        print("\n" + "="*70)
        print("PROTOCOL V2 - MULTI-MODALITY TEST SUITE")
        print("="*70)
        
        self.test_vision_modality()
        self.test_text_modality()
        self.test_mixed_modality()
        self.test_high_dimensional_input()
        self.test_time_series_modality()
        
        print("\n" + "="*70)
        print(f"RESULTS: {self.results['tests_passed']} PASSED | {self.results['tests_failed']} FAILED")
        print("="*70)
        
        # Save results
        results_dir = Path(__file__).parent.parent / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / 'multimodality_test_results.json'
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        return self.results


if __name__ == '__main__':
    tester = MultiModalityTester(device='cpu')
    results = tester.run_all()
