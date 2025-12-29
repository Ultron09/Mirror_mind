"""
Protocol v2: Simple Direct Test Runner
=======================================
Runs all tests directly without subprocess complexity.
"""

import sys
import os
from pathlib import Path

# Set encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add tests directory to path
test_dir = Path(__file__).parent / 'tests'
sys.path.insert(0, str(test_dir))

def run_single_test(test_module_name):
    """Import and run a single test module."""
    try:
        # Import the module
        test_module = __import__(test_module_name)
        
        # The module's main() will be called by __main__ guard
        print(f"✓ {test_module_name.replace('_', ' ').title()} executed")
        return True
    except Exception as e:
        print(f"✗ {test_module_name.replace('_', ' ').title()} failed: {str(e)[:100]}")
        return False

def main():
    """Run all test suites."""
    test_suites = [
        'test_integration',
        'test_usability',
        'test_baselines',
        'test_multimodality',
        'test_memory_stress',
        'test_adaptation_extremes',
        'test_survival_scenarios'
    ]
    
    print("\n" + "="*70)
    print("PROTOCOL V2 - TEST EXECUTION")
    print("="*70)
    
    results = {}
    for suite in test_suites:
        print(f"\nRunning: {suite.replace('_', ' ').title()}...")
        try:
            # Change to test directory and run directly
            os.chdir(test_dir)
            
            # Execute the test script directly
            test_file = test_dir / f"{suite}.py"
            with open(test_file, 'r') as f:
                code = f.read()
            
            # Execute in isolated namespace
            namespace = {'__name__': '__main__', '__file__': str(test_file)}
            exec(code, namespace)
            
            results[suite] = True
        except Exception as e:
            print(f"Error: {str(e)[:100]}")
            results[suite] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    
    for suite, success in results.items():
        symbol = "✓" if success else "✗"
        print(f"  {symbol} {suite.replace('_', ' ').title()}")
    
    print(f"\nTotal: {passed}/{len(results)} completed successfully")
    
    # Check results files
    results_dir = Path(__file__).parent / 'results'
    result_files = list(results_dir.glob('*_results.json'))
    
    print(f"\nGenerated {len(result_files)} result files:")
    for rf in sorted(result_files):
        file_size = rf.stat().st_size
        print(f"  - {rf.name} ({file_size} bytes)")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    main()
