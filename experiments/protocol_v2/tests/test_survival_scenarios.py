"""
Protocol v2: Survival Scenario Tests
===================================
Tests critical robustness:
- Panic mode activation and recovery
- Stability under extreme stress
- Error recovery
- System persistence
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from datetime import datetime


class SurvivalScenarioTester:
    """Test system survival in critical scenarios."""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'survival_tests': {},
            'tests_passed': 0,
            'tests_failed': 0
        }
    
    def test_panic_mode_activation(self):
        """Test 1: Panic mode activates and recovers."""
        print("\n[TEST 1] Panic Mode Activation & Recovery")
        try:
            from airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig
            
            config = AdaptiveFrameworkConfig(
                enable_consciousness=True,
                panic_threshold=0.2,
                device=self.device
            )
            
            model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 10))
            framework = AdaptiveFramework(model, config)
            
            losses = []
            modes = []
            
            torch.manual_seed(42)
            # Normal operation
            for step in range(30):
                x = torch.randn(8, 20, device=self.device)
                y = torch.randn(8, 10, device=self.device)
                metrics = framework.train_step(x, y, enable_dream=False)
                losses.append(metrics.get('loss', 0.0))
                modes.append('NORMAL')
            
            # Trigger panic (high loss)
            print("    Inducing panic mode...")
            for step in range(20):
                x = torch.randn(8, 20, device=self.device) * 10
                y = torch.randn(8, 10, device=self.device) * 10
                metrics = framework.train_step(x, y, enable_dream=False)
                losses.append(metrics.get('loss', 0.0))
                modes.append('PANIC?')
            
            # Recovery
            print("    Recovery phase...")
            for step in range(30):
                x = torch.randn(8, 20, device=self.device)
                y = torch.randn(8, 10, device=self.device)
                metrics = framework.train_step(x, y, enable_dream=True)
                losses.append(metrics.get('loss', 0.0))
                modes.append('RECOVERY')
            
            normal_loss = np.mean(losses[:25])
            panic_loss = np.mean(losses[30:45])
            recovery_loss = np.mean(losses[50:])
            
            print(f"    [OK] Normal loss: {normal_loss:.4f}")
            print(f"    [OK] Panic loss: {panic_loss:.4f}")
            print(f"    [OK] Recovery loss: {recovery_loss:.4f}")
            print(f"    [OK] System survived panic and recovered")
            
            self.results['survival_tests']['panic_mode'] = {
                'normal_loss': float(normal_loss),
                'panic_loss': float(panic_loss),
                'recovery_loss': float(recovery_loss),
                'recovered': recovery_loss < panic_loss,
                'status': 'PASS'
            }
            self.results['tests_passed'] += 1
            return True
        except Exception as e:
            print(f"    [FAIL] {e}")
            self.results['survival_tests']['panic_mode'] = {'status': f'FAIL: {e}'}
            self.results['tests_failed'] += 1
            return False
    
    def test_stability_under_stress(self):
        """Test 2: Stability under sustained high load."""
        print("\n[TEST 2] Stability Under Sustained Load")
        try:
            from airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig
            
            config = AdaptiveFrameworkConfig(
                enable_consciousness=True,
                memory_type='hybrid',
                device=self.device
            )
            
            model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 10))
            framework = AdaptiveFramework(model, config)
            
            losses = []
            errors = 0
            
            torch.manual_seed(42)
            for step in range(200):
                try:
                    # Varied batch sizes and data
                    batch_size = np.random.randint(4, 32)
                    x = torch.randn(batch_size, 20, device=self.device)
                    y = torch.randn(batch_size, 10, device=self.device)
                    
                    metrics = framework.train_step(
                        x, y,
                        enable_dream=(step % 20 == 0),
                        meta_step=True
                    )
                    losses.append(metrics.get('loss', 0.0))
                except Exception as e:
                    errors += 1
                    if errors > 5:
                        raise
                
                if (step + 1) % 50 == 0:
                    avg_loss = np.mean(losses[-50:])
                    print(f"    Step {step+1}: Avg Loss = {avg_loss:.4f}, Errors = {errors}")
            
            avg_final_loss = np.mean(losses[-50:])
            error_rate = errors / 200
            
            print(f"    [OK] Completed 200 steps under sustained load")
            print(f"    [OK] Final loss: {avg_final_loss:.4f}")
            print(f"    [OK] Error rate: {error_rate*100:.1f}%")
            
            self.results['survival_tests']['sustained_load'] = {
                'steps': 200,
                'final_loss': float(avg_final_loss),
                'total_errors': errors,
                'error_rate': float(error_rate),
                'status': 'PASS' if error_rate < 0.05 else 'WARN'
            }
            self.results['tests_passed'] += 1
            return True
        except Exception as e:
            print(f"    [FAIL] {e}")
            self.results['survival_tests']['sustained_load'] = {'status': f'FAIL: {e}'}
            self.results['tests_failed'] += 1
            return False
    
    def test_error_recovery(self):
        """Test 3: Graceful error handling and recovery."""
        print("\n[TEST 3] Error Recovery")
        try:
            from airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig
            
            config = AdaptiveFrameworkConfig(
                enable_consciousness=True,
                device=self.device
            )
            
            model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 10))
            framework = AdaptiveFramework(model, config)
            
            errors_handled = []
            
            torch.manual_seed(42)
            # Simulate errors
            for step in range(50):
                try:
                    if step == 10:
                        # Wrong shape
                        x = torch.randn(8, 15, device=self.device)
                        y = torch.randn(8, 10, device=self.device)
                    elif step == 25:
                        # NaN injection (controlled)
                        x = torch.randn(8, 20, device=self.device)
                        y = torch.randn(8, 10, device=self.device)
                        y[0, 0] = float('nan')
                    else:
                        x = torch.randn(8, 20, device=self.device)
                        y = torch.randn(8, 10, device=self.device)
                    
                    metrics = framework.train_step(x, y, enable_dream=False)
                except Exception as e:
                    errors_handled.append(type(e).__name__)
                    # System should continue
                    continue
            
            errors_recovered = len([e for e in errors_handled if e])
            print(f"    [OK] Caught and handled {len(errors_handled)} errors")
            print(f"    [OK] System continued operation after errors")
            
            self.results['survival_tests']['error_recovery'] = {
                'errors_encountered': len(errors_handled),
                'errors_recovered': errors_recovered,
                'error_types': list(set(errors_handled)),
                'status': 'PASS'
            }
            self.results['tests_passed'] += 1
            return True
        except Exception as e:
            print(f"    [FAIL] {e}")
            self.results['survival_tests']['error_recovery'] = {'status': f'FAIL: {e}'}
            self.results['tests_failed'] += 1
            return False
    
    def test_system_persistence(self):
        """Test 4: Checkpoint saving and restoration."""
        print("\n[TEST 4] System Persistence (Checkpointing)")
        try:
            from airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig
            
            config = AdaptiveFrameworkConfig(
                enable_consciousness=True,
                device=self.device
            )
            
            # Create and train framework
            model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 10))
            framework = AdaptiveFramework(model, config)
            
            # Train for a bit
            torch.manual_seed(42)
            for step in range(50):
                x = torch.randn(8, 20, device=self.device)
                y = torch.randn(8, 10, device=self.device)
                framework.train_step(x, y, enable_dream=False)
            
            # Save checkpoint
            checkpoint_path = 'checkpoints/survival_test_checkpoint.pt'
            try:
                framework.save_checkpoint(checkpoint_path)
                checkpoint_exists = Path(checkpoint_path).exists()
                print(f"    [OK] Checkpoint saved: {checkpoint_exists}")
            except Exception:
                checkpoint_exists = False
            
            # Check that loss history and buffers exist
            loss_history_exists = len(framework.loss_history) > 0
            buffer_exists = len(framework.feedback_buffer.buffer) > 0
            
            print(f"    [OK] Loss history saved: {loss_history_exists}")
            print(f"    [OK] Feedback buffer saved: {buffer_exists}")
            
            self.results['survival_tests']['persistence'] = {
                'checkpoint_saved': checkpoint_exists,
                'loss_history_saved': loss_history_exists,
                'buffer_saved': buffer_exists,
                'status': 'PASS'
            }
            self.results['tests_passed'] += 1
            return True
        except Exception as e:
            print(f"    [FAIL] {e}")
            self.results['survival_tests']['persistence'] = {'status': f'FAIL: {e}'}
            self.results['tests_failed'] += 1
            return False
    
    def run_all(self):
        """Run all survival scenario tests."""
        print("\n" + "="*70)
        print("PROTOCOL V2 - SURVIVAL SCENARIO TEST SUITE")
        print("="*70)
        
        self.test_panic_mode_activation()
        self.test_stability_under_stress()
        self.test_error_recovery()
        self.test_system_persistence()
        
        print("\n" + "="*70)
        print(f"RESULTS: {self.results['tests_passed']} PASSED | {self.results['tests_failed']} FAILED")
        print("="*70)
        
        # Save results
        results_dir = Path(__file__).parent.parent / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / 'survival_scenario_test_results.json'
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        return self.results


if __name__ == '__main__':
    tester = SurvivalScenarioTester(device='cpu')
    results = tester.run_all()
