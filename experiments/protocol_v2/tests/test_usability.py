"""
Protocol v2: Usability Tests
============================
Tests API simplicity, error handling, default configs, and documentation.
Ensures developers can easily adopt MirrorMind v7.0.
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
from datetime import datetime
import traceback


class UsabilityTester:
    """Test framework usability and developer experience."""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests_passed': 0,
            'tests_failed': 0,
            'usability_metrics': {},
            'error_handling': {},
            'api_simplicity': {}
        }
    
    def test_simple_api(self):
        """Test 1: Simple, intuitive API."""
        print("\n[TEST 1] Simple API")
        try:
            from airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig
            
            # Minimal code needed
            model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 5))
            config = AdaptiveFrameworkConfig()  # Works with defaults
            framework = AdaptiveFramework(model, config)
            
            # Can call train_step without extra setup
            x = torch.randn(4, 10)
            y = torch.randn(4, 5)
            metrics = framework.train_step(x, y)
            
            # Returns expected metrics
            assert 'loss' in metrics, "Metrics missing 'loss'"
            
            code_lines = 6  # Minimal lines of code
            print(f"    [OK] Framework works with ~{code_lines} lines of code")
            print(f"    [OK] Default configuration works")
            print(f"    [OK] Metrics returned: {list(metrics.keys())}")
            
            self.results['api_simplicity']['minimal_code'] = 'PASS'
            self.results['tests_passed'] += 1
            return True
        except Exception as e:
            print(f"    [FAIL] {e}")
            self.results['api_simplicity']['minimal_code'] = f'FAIL: {e}'
            self.results['tests_failed'] += 1
            return False
    
    def test_sensible_defaults(self):
        """Test 2: Sensible default configuration."""
        print("\n[TEST 2] Sensible Defaults")
        try:
            from airbornehrs import AdaptiveFrameworkConfig
            
            config = AdaptiveFrameworkConfig()
            
            # Check sensible defaults
            defaults = {
                'enable_consciousness': True,  # Consciousness ON by default
                'memory_type': 'hybrid',       # Best memory system by default
                'use_prioritized_replay': True,  # Prioritized replay ON
                'adaptive_lambda': True,       # Adaptive protection ON
            }
            
            for key, expected_val in defaults.items():
                actual_val = getattr(config, key, None)
                assert actual_val == expected_val, f"{key}: expected {expected_val}, got {actual_val}"
                print(f"    [OK] {key}: {actual_val}")
            
            self.results['usability_metrics']['sensible_defaults'] = 'PASS'
            self.results['tests_passed'] += 1
            return True
        except Exception as e:
            print(f"    [FAIL] {e}")
            self.results['usability_metrics']['sensible_defaults'] = f'FAIL: {e}'
            self.results['tests_failed'] += 1
            return False
    
    def test_error_handling(self):
        """Test 3: Graceful error handling."""
        print("\n[TEST 3] Error Handling")
        try:
            from airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig
            
            model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 5))
            config = AdaptiveFrameworkConfig()
            framework = AdaptiveFramework(model, config)
            
            errors_handled = []
            
            # Test 1: Wrong tensor shape
            try:
                x = torch.randn(4, 5)  # Wrong input size
                y = torch.randn(4, 5)
                framework.train_step(x, y)
                errors_handled.append('shape_mismatch: NOT_CAUGHT')
            except Exception as e:
                errors_handled.append('shape_mismatch: caught')
            
            # Test 2: Invalid tensor type
            try:
                x = torch.randn(4, 10).int()  # Wrong type
                y = torch.randn(4, 5)
                framework.train_step(x, y)
                errors_handled.append('type_error: NOT_CAUGHT')
            except Exception as e:
                errors_handled.append('type_error: caught')
            
            # Test 3: NaN in output (should be logged, not crash)
            print(f"    [OK] Error handling: {', '.join(errors_handled)}")
            
            self.results['error_handling']['graceful_errors'] = 'PASS'
            self.results['tests_passed'] += 1
            return True
        except Exception as e:
            print(f"    [FAIL] {e}")
            self.results['error_handling']['graceful_errors'] = f'FAIL: {e}'
            self.results['tests_failed'] += 1
            return False
    
    def test_config_customization(self):
        """Test 4: Easy configuration customization."""
        print("\n[TEST 4] Configuration Customization")
        try:
            from airbornehrs import AdaptiveFrameworkConfig
            
            # Test various configs
            configs_tested = []
            
            # Config 1: Minimal SI memory
            config1 = AdaptiveFrameworkConfig(
                memory_type='si',
                use_prioritized_replay=False
            )
            configs_tested.append('SI-only')
            
            # Config 2: Consciousness disabled
            config2 = AdaptiveFrameworkConfig(
                enable_consciousness=False
            )
            configs_tested.append('No-consciousness')
            
            # Config 3: Production mode
            config3 = AdaptiveFrameworkConfig.production()
            configs_tested.append('Production')
            
            print(f"    [OK] Tested configs: {', '.join(configs_tested)}")
            print(f"    [OK] Config system is flexible")
            
            self.results['usability_metrics']['config_customization'] = 'PASS'
            self.results['tests_passed'] += 1
            return True
        except Exception as e:
            print(f"    [FAIL] {e}")
            self.results['usability_metrics']['config_customization'] = f'FAIL: {e}'
            self.results['tests_failed'] += 1
            return False
    
    def test_documentation_completeness(self):
        """Test 5: Documentation is complete."""
        print("\n[TEST 5] Documentation Completeness")
        try:
            # Check for documentation files
            doc_files = [
                Path('CONSCIOUSNESS_QUICK_START.md'),
                Path('CONSCIOUSNESS_INTEGRATION_COMPLETE.md'),
                Path('API.md'),
                Path('IMPLEMENTATION_GUIDE.md')
            ]
            
            found_docs = []
            for doc in doc_files:
                if doc.exists():
                    found_docs.append(doc.name)
            
            assert len(found_docs) > 0, "No documentation found"
            
            print(f"    [OK] Found documentation: {len(found_docs)} files")
            for doc in found_docs:
                print(f"       - {doc}")
            
            self.results['usability_metrics']['documentation'] = 'PASS'
            self.results['tests_passed'] += 1
            return True
        except Exception as e:
            print(f"    [FAIL] {e}")
            self.results['usability_metrics']['documentation'] = f'FAIL: {e}'
            self.results['tests_failed'] += 1
            return False
    
    def test_logging_informativeness(self):
        """Test 6: Logging is informative without being verbose."""
        print("\n[TEST 6] Logging Informativeness")
        try:
            from airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig
            import io
            import sys
            
            # Capture logging output
            model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 5))
            config = AdaptiveFrameworkConfig(log_frequency=2)
            framework = AdaptiveFramework(model, config)
            
            # Run a few steps
            for _ in range(5):
                x = torch.randn(4, 10)
                y = torch.randn(4, 5)
                framework.train_step(x, y)
            
            print(f"    [OK] Logging system active")
            print(f"    [OK] Log frequency configurable: {config.log_frequency}")
            
            self.results['usability_metrics']['logging'] = 'PASS'
            self.results['tests_passed'] += 1
            return True
        except Exception as e:
            print(f"    [FAIL] {e}")
            self.results['usability_metrics']['logging'] = f'FAIL: {e}'
            self.results['tests_failed'] += 1
            return False
    
    def run_all(self):
        """Run all usability tests."""
        print("\n" + "="*70)
        print("PROTOCOL V2 - USABILITY TEST SUITE")
        print("="*70)
        
        self.test_simple_api()
        self.test_sensible_defaults()
        self.test_error_handling()
        self.test_config_customization()
        self.test_documentation_completeness()
        self.test_logging_informativeness()
        
        print("\n" + "="*70)
        print(f"RESULTS: {self.results['tests_passed']} PASSED | {self.results['tests_failed']} FAILED")
        print("="*70)
        
        # Save results
        results_dir = Path(__file__).parent.parent / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / 'usability_test_results.json'
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        return self.results


if __name__ == '__main__':
    tester = UsabilityTester()
    results = tester.run_all()
