"""
Protocol v2: Master Test Runner
================================
Executes all test suites sequentially and aggregates results.
"""

import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict


class MasterTestRunner:
    """Run all protocol_v2 tests and aggregate results."""
    
    def __init__(self, test_dir='experiments/protocol_v2/tests',
                 results_dir='experiments/protocol_v2/results'):
        self.test_dir = Path(test_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_files = [
            'test_integration.py',
            'test_usability.py',
            'test_baselines.py',
            'test_multimodality.py',
            'test_memory_stress.py',
            'test_adaptation_extremes.py',
            'test_survival_scenarios.py'
        ]
        
        self.test_results = {}
        self.summary = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_results': {}
        }
    
    def run_test_suite(self, test_file):
        """Run a single test suite."""
        test_path = self.test_dir / test_file
        test_name = test_file.replace('.py', '')
        
        print(f"\n{'='*70}")
        print(f"RUNNING: {test_name.replace('_', ' ').upper()}")
        print(f"{'='*70}")
        
        try:
            # Run the test file
            result = subprocess.run(
                [sys.executable, str(test_path)],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            
            # Try to load results from JSON
            result_file = self.results_dir / f'{test_name}_results.json'
            if result_file.exists():
                with open(result_file, 'r') as f:
                    test_result = json.load(f)
                    passed = test_result.get('tests_passed', 0)
                    failed = test_result.get('tests_failed', 0)
                    
                    self.summary['test_results'][test_name] = {
                        'status': 'PASSED' if failed == 0 else 'PARTIAL',
                        'passed': passed,
                        'failed': failed,
                        'total': passed + failed
                    }
                    
                    self.summary['total_tests'] += passed + failed
                    self.summary['passed_tests'] += passed
                    self.summary['failed_tests'] += failed
                    
                    print(f"\n✓ Test suite completed: {passed}/{passed + failed} tests passed")
            else:
                print(f"\n⚠ Warning: No results file found for {test_name}")
                self.summary['test_results'][test_name] = {
                    'status': 'UNKNOWN',
                    'error': 'Results file not found'
                }
            
            return True
            
        except subprocess.TimeoutExpired:
            print(f"\n✗ Test suite TIMEOUT after 300 seconds")
            self.summary['test_results'][test_name] = {
                'status': 'TIMEOUT',
                'error': 'Test execution exceeded 300 second timeout'
            }
            return False
        
        except Exception as e:
            print(f"\n✗ Test suite FAILED with error: {str(e)}")
            self.summary['test_results'][test_name] = {
                'status': 'ERROR',
                'error': str(e)
            }
            return False
    
    def run_all(self):
        """Run all test suites."""
        print("\n" + "="*70)
        print("PROTOCOL V2 - MASTER TEST RUNNER")
        print("="*70)
        print(f"Test directory: {self.test_dir}")
        print(f"Results directory: {self.results_dir}")
        print(f"Starting: {self.summary['timestamp']}")
        print(f"Tests to run: {len(self.test_files)}")
        
        successful = 0
        failed = 0
        
        for test_file in self.test_files:
            test_path = self.test_dir / test_file
            
            if not test_path.exists():
                print(f"\n✗ Test file not found: {test_file}")
                failed += 1
                continue
            
            if self.run_test_suite(test_file):
                successful += 1
            else:
                failed += 1
        
        return successful, failed
    
    def save_summary(self):
        """Save aggregated summary to JSON."""
        self.summary['execution_summary'] = {
            'total_suites': len(self.test_files),
            'execution_time': datetime.now().isoformat(),
            'success_rate': f"{(self.summary['passed_tests'] / max(self.summary['total_tests'], 1) * 100):.1f}%"
        }
        
        summary_file = self.results_dir / 'protocol_v2_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(self.summary, f, indent=2)
        
        print(f"\nSummary saved to: {summary_file}")
        return summary_file
    
    def print_summary(self):
        """Print execution summary."""
        print("\n" + "="*70)
        print("PROTOCOL V2 - TEST EXECUTION SUMMARY")
        print("="*70)
        print(f"Total Tests: {self.summary['total_tests']}")
        print(f"Passed: {self.summary['passed_tests']}")
        print(f"Failed: {self.summary['failed_tests']}")
        
        if self.summary['total_tests'] > 0:
            pass_rate = (self.summary['passed_tests'] / self.summary['total_tests']) * 100
            print(f"Pass Rate: {pass_rate:.1f}%")
        
        print("\nTest Suite Results:")
        for suite_name, result in self.summary['test_results'].items():
            status = result.get('status', 'UNKNOWN')
            if status == 'PASSED':
                symbol = "✓"
            elif status == 'PARTIAL':
                symbol = "⚠"
            elif status == 'ERROR':
                symbol = "✗"
            else:
                symbol = "?"
            
            suite_display = suite_name.replace('_', ' ').title()
            if 'passed' in result and 'failed' in result:
                print(f"  {symbol} {suite_display}: {result['passed']}/{result['total']} passed")
            else:
                print(f"  {symbol} {suite_display}: {status}")
        
        print("\n" + "="*70)


def main():
    """Main entry point."""
    runner = MasterTestRunner()
    
    # Run all tests
    successful, failed = runner.run_all()
    
    # Save and print summary
    runner.save_summary()
    runner.print_summary()
    
    # Exit with appropriate code
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
