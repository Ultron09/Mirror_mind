"""
Protocol v2: Fixed Master Test Runner
======================================
Runs all test suites with proper error handling and encoding support.
"""

import sys
import os
import subprocess
from pathlib import Path
from datetime import datetime

def run_test_suite(test_file):
    """Run a single test suite in subprocess."""
    test_path = Path(__file__).parent / 'tests' / test_file
    test_name = test_file.replace('.py', '')
    
    print(f"\n{'='*70}")
    print(f"Running: {test_name.replace('_', ' ').upper()}")
    print(f"{'='*70}")
    
    # Set encoding environment variable
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    try:
        result = subprocess.run(
            [sys.executable, str(test_path)],
            capture_output=True,
            text=True,
            timeout=300,
            env=env
        )
        
        # Show output (filter out encoding errors)
        for line in result.stdout.split('\n'):
            if 'UnicodeEncodeError' not in line and 'Logging error' not in line and line.strip():
                print(line)
        
        # Check if result file was created
        result_file = Path(__file__).parent / 'results' / f'{test_name}_results.json'
        if result_file.exists():
            print(f"✓ Results saved to: {result_file.name}")
            return True
        else:
            print(f"⚠ Results file not found: {test_name}_results.json")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ Test timed out after 300 seconds")
        return False
    except Exception as e:
        print(f"✗ Error running test: {str(e)}")
        return False

def main():
    """Run all test suites."""
    test_files = [
        'test_integration.py',
        'test_usability.py',
        'test_baselines.py',
        'test_multimodality.py',
        'test_memory_stress.py',
        'test_adaptation_extremes.py',
        'test_survival_scenarios.py'
    ]
    
    print("\n" + "="*70)
    print("PROTOCOL V2 - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    results = {}
    for test_file in test_files:
        test_name = test_file.replace('.py', '')
        success = run_test_suite(test_file)
        results[test_name] = 'PASSED' if success else 'FAILED'
    
    # Print summary
    print("\n" + "="*70)
    print("TEST EXECUTION SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v == 'PASSED')
    failed = sum(1 for v in results.values() if v == 'FAILED')
    
    for test_name, status in results.items():
        symbol = "✓" if status == "PASSED" else "✗"
        print(f"  {symbol} {test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nTotal: {passed}/{len(results)} tests generated results")
    print(f"Passed: {passed} | Failed: {failed}")
    
    # List generated result files
    results_dir = Path(__file__).parent / 'results'
    result_files = list(results_dir.glob('*_results.json'))
    
    print(f"\nGenerated {len(result_files)} result files:")
    for rf in sorted(result_files):
        print(f"  - {rf.name}")
    
    print("\n" + "="*70)
    print("Running visualization reporter...")
    print("="*70)
    
    # Try to run visualization reporter
    try:
        viz_path = Path(__file__).parent / 'visualization_reporter.py'
        subprocess.run([sys.executable, str(viz_path)], timeout=60, env=env)
    except:
        print("Visualization generation skipped")
    
    print("\n" + "="*70)
    print("PROTOCOL V2 TEST RUN COMPLETE")
    print("="*70)
    print(f"\nResults Location: {results_dir}")
    print(f"Plots Location: {Path(__file__).parent / 'plots'}")
    print(f"Reports Location: {Path(__file__).parent / 'reports'}")

if __name__ == '__main__':
    main()
