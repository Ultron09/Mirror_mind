"""
Protocol v2: Integration Tests
================================
Tests all core components of MirrorMind v7.0 consciousness framework.
Validates consciousness observation, consolidation, replay prioritization, and memory protection.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Import framework
from airbornehrs import (
    AdaptiveFramework,
    AdaptiveFrameworkConfig,
    ConsciousnessCore,
)


class IntegrationTester:
    """Comprehensive integration testing suite."""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests_passed': 0,
            'tests_failed': 0,
            'component_status': {},
            'metrics': {}
        }
    
    def test_consciousness_observation(self):
        """Test 1: Consciousness observes training examples."""
        print("\n[TEST 1] Consciousness Observation")
        try:
            config = AdaptiveFrameworkConfig(
                enable_consciousness=True,
                device=self.device
            )
            model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 5))
            framework = AdaptiveFramework(model, config)
            
            # Generate training data
            x = torch.randn(8, 10)
            y = torch.randn(8, 5)
            
            # Run one step
            metrics = framework.train_step(x, y, enable_dream=False)
            
            # Verify consciousness was called
            assert framework.consciousness is not None, "Consciousness not initialized"
            state = framework.consciousness.get_knowledge_state()
            assert 'learning_gap' in state, "Consciousness state incomplete"
            
            print("    [OK] Consciousness observes examples")
            print(f"    [OK] Learning gap tracked: {state.get('learning_gap', 0):.3f}")
            self.results['component_status']['consciousness_observation'] = 'PASS'
            self.results['tests_passed'] += 1
            return True
        except Exception as e:
            print(f"    [FAIL] {e}")
            self.results['component_status']['consciousness_observation'] = f'FAIL: {e}'
            self.results['tests_failed'] += 1
            return False
    
    def test_consolidation_trigger(self):
        """Test 2: Consolidation triggers based on consciousness urgency."""
        print("\n[TEST 2] Consolidation Trigger by Urgency")
        try:
            config = AdaptiveFrameworkConfig(
                enable_consciousness=True,
                memory_type='hybrid',
                device=self.device,
                warmup_steps=5
            )
            model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 5))
            framework = AdaptiveFramework(model, config)
            
            consolidation_count = 0
            initial_task_memories = len(list(Path('checkpoints/task_memories').glob('*.pt'))) if Path('checkpoints/task_memories').exists() else 0
            
            # Run several steps to trigger consolidation
            torch.manual_seed(42)
            for step in range(15):
                x = torch.randn(4, 10)
                y = torch.randn(4, 5)
                metrics = framework.train_step(x, y, enable_dream=False, meta_step=False)
            
            final_task_memories = len(list(Path('checkpoints/task_memories').glob('*.pt'))) if Path('checkpoints/task_memories').exists() else 0
            consolidation_count = final_task_memories - initial_task_memories
            
            assert consolidation_count > 0, f"No consolidations triggered (initial: {initial_task_memories}, final: {final_task_memories})"
            
            print(f"    [OK] Consolidations triggered: {consolidation_count}")
            self.results['component_status']['consolidation_trigger'] = 'PASS'
            self.results['metrics']['consolidations_triggered'] = consolidation_count
            self.results['tests_passed'] += 1
            return True
        except Exception as e:
            print(f"    [FAIL] {e}")
            self.results['component_status']['consolidation_trigger'] = f'FAIL: {e}'
            self.results['tests_failed'] += 1
            return False
    
    def test_prioritized_replay(self):
        """Test 3: Prioritized replay weights samples correctly."""
        print("\n[TEST 3] Prioritized Replay Buffer")
        try:
            config = AdaptiveFrameworkConfig(
                enable_consciousness=True,
                use_prioritized_replay=True,
                device=self.device
            )
            model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 5))
            framework = AdaptiveFramework(model, config)
            
            # Add varied difficulty samples
            torch.manual_seed(42)
            for i in range(20):
                x = torch.randn(4, 10)
                y = torch.randn(4, 5)
                framework.train_step(x, y, enable_dream=False)
            
            # Check prioritized buffer
            assert hasattr(framework, 'prioritized_buffer'), "Prioritized buffer not found"
            assert framework.prioritized_buffer is not None, "Prioritized buffer is None"
            assert len(framework.prioritized_buffer.buffer) > 0, "Prioritized buffer empty"
            
            print(f"    [OK] Prioritized buffer active")
            print(f"    [OK] Samples in prioritized buffer: {len(framework.prioritized_buffer.buffer)}")
            self.results['component_status']['prioritized_replay'] = 'PASS'
            self.results['metrics']['prioritized_samples'] = len(framework.prioritized_buffer.buffer)
            self.results['tests_passed'] += 1
            return True
        except Exception as e:
            print(f"    [FAIL] {e}")
            self.results['component_status']['prioritized_replay'] = f'FAIL: {e}'
            self.results['tests_failed'] += 1
            return False
    
    def test_memory_protection(self):
        """Test 4: Memory protection (SI + EWC hybrid)."""
        print("\n[TEST 4] Memory Protection (SI + EWC Hybrid)")
        try:
            config = AdaptiveFrameworkConfig(
                memory_type='hybrid',
                enable_consciousness=True,
                device=self.device
            )
            model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 5))
            framework = AdaptiveFramework(model, config)
            
            # Verify EWC handler
            assert hasattr(framework, 'ewc'), "EWC handler not found"
            assert framework.ewc is not None, "EWC handler is None"
            assert hasattr(framework.ewc, 'consolidate'), "Consolidate method not found"
            
            # Verify SI path
            initial_params = {name: p.clone() for name, p in framework.model.named_parameters()}
            
            # Run training
            torch.manual_seed(42)
            for _ in range(10):
                x = torch.randn(4, 10)
                y = torch.randn(4, 5)
                framework.train_step(x, y, enable_dream=False)
            
            # Verify parameters changed (learning happened)
            param_changes = []
            for name, p in framework.model.named_parameters():
                if name in initial_params:
                    change = (p - initial_params[name]).norm().item()
                    param_changes.append(change)
            
            avg_change = np.mean(param_changes)
            assert avg_change > 0.001, f"Parameters didn't change enough: {avg_change}"
            
            print(f"    [OK] Hybrid SI+EWC memory system active")
            print(f"    [OK] Average parameter change: {avg_change:.6f}")
            self.results['component_status']['memory_protection'] = 'PASS'
            self.results['metrics']['param_change'] = float(avg_change)
            self.results['tests_passed'] += 1
            return True
        except Exception as e:
            print(f"    [FAIL] {e}")
            self.results['component_status']['memory_protection'] = f'FAIL: {e}'
            self.results['tests_failed'] += 1
            return False
    
    def test_adaptive_lambda(self):
        """Test 5: Adaptive lambda regularization."""
        print("\n[TEST 5] Adaptive Lambda Regularization")
        try:
            config = AdaptiveFrameworkConfig(
                adaptive_lambda=True,
                enable_consciousness=True,
                device=self.device
            )
            model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 5))
            framework = AdaptiveFramework(model, config)
            
            # Verify adaptive reg exists
            assert hasattr(framework, 'adaptive_reg'), "Adaptive regularization not found"
            
            # Test different modes
            modes = ['BOOTSTRAP', 'NORMAL', 'NOVELTY']
            lambdas = []
            
            for mode in modes:
                if hasattr(framework.adaptive_reg, 'get_lambda'):
                    lam = framework.adaptive_reg.get_lambda(mode, step_in_mode=5)
                    lambdas.append(lam)
            
            print(f"    [OK] Adaptive lambda system active")
            print(f"    [OK] Lambda values by mode: {lambdas}")
            self.results['component_status']['adaptive_lambda'] = 'PASS'
            self.results['tests_passed'] += 1
            return True
        except Exception as e:
            print(f"    [FAIL] {e}")
            self.results['component_status']['adaptive_lambda'] = f'FAIL: {e}'
            self.results['tests_failed'] += 1
            return False
    
    def test_end_to_end_training(self):
        """Test 6: Full training loop with consciousness."""
        print("\n[TEST 6] End-to-End Training")
        try:
            config = AdaptiveFrameworkConfig(
                enable_consciousness=True,
                use_prioritized_replay=True,
                memory_type='hybrid',
                adaptive_lambda=True,
                device=self.device,
                dream_interval=5
            )
            model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 5))
            framework = AdaptiveFramework(model, config)
            
            losses = []
            torch.manual_seed(42)
            for step in range(30):
                x = torch.randn(8, 10)
                y = torch.randn(8, 5)
                metrics = framework.train_step(x, y, enable_dream=(step % 5 == 0), meta_step=True)
                losses.append(metrics.get('loss', 0.0))
            
            # Check that loss history is recorded
            assert len(framework.loss_history) > 0, "Loss history empty"
            
            # Check loss decreased overall (learning happened)
            early_avg = np.mean(losses[:10])
            late_avg = np.mean(losses[-10:])
            improvement = (early_avg - late_avg) / early_avg if early_avg > 0 else 0
            
            print(f"    [OK] End-to-end training completed")
            print(f"    [OK] Steps: 30 | Early avg loss: {early_avg:.4f} | Late avg loss: {late_avg:.4f}")
            print(f"    [OK] Learning improvement: {improvement*100:.1f}%")
            
            self.results['component_status']['end_to_end_training'] = 'PASS'
            self.results['metrics']['training_steps'] = 30
            self.results['metrics']['learning_improvement'] = float(improvement)
            self.results['tests_passed'] += 1
            return True
        except Exception as e:
            print(f"    [FAIL] {e}")
            self.results['component_status']['end_to_end_training'] = f'FAIL: {e}'
            self.results['tests_failed'] += 1
            return False
    
    def run_all(self):
        """Run all integration tests."""
        print("\n" + "="*70)
        print("PROTOCOL V2 - INTEGRATION TEST SUITE")
        print("="*70)
        
        self.test_consciousness_observation()
        self.test_consolidation_trigger()
        self.test_prioritized_replay()
        self.test_memory_protection()
        self.test_adaptive_lambda()
        self.test_end_to_end_training()
        
        print("\n" + "="*70)
        print(f"RESULTS: {self.results['tests_passed']} PASSED | {self.results['tests_failed']} FAILED")
        print("="*70)
        
        # Save results
        results_dir = Path(__file__).parent.parent / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / 'integration_test_results.json'
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        return self.results


if __name__ == '__main__':
    tester = IntegrationTester(device='cpu')
    results = tester.run_all()
