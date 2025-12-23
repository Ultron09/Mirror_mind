"""
Protocol v2: Quick-Start Guide
===============================
Run all tests and generate publication-ready reports.
"""

import os
import sys
import json
from pathlib import Path

# Ensure we can import the framework
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def quick_start():
    """Run complete protocol v2 pipeline."""
    
    print("\n" + "="*80)
    print(" "*20 + "PROTOCOL V2 - QUICK START")
    print("="*80)
    
    protocol_dir = Path(__file__).parent
    test_dir = protocol_dir / 'tests'
    
    print("\n[1/3] Running test suites...")
    print("-" * 80)
    
    # Import and run individual test suites
    test_modules = [
        'test_integration',
        'test_usability',
        'test_baselines',
        'test_multimodality',
        'test_memory_stress',
        'test_adaptation_extremes',
        'test_survival_scenarios'
    ]
    
    results = {}
    
    for test_module in test_modules:
        try:
            # Import the test module
            module_path = test_dir / f'{test_module}.py'
            
            # Read and execute
            with open(module_path, 'r') as f:
                code = f.read()
            
            # Create module namespace
            module_namespace = {'__name__': test_module}
            exec(code, module_namespace)
            
            # Get the tester class
            class_name = ''.join(word.capitalize() for word in test_module.split('_'))
            if 'Tester' not in class_name:
                if 'baseline' in test_module:
                    class_name = 'BaselineComparator'
                else:
                    class_name = class_name + 'Tester'
            
            if class_name in module_namespace:
                tester = module_namespace[class_name]()
                print(f"\n✓ {test_module.replace('_', ' ').title()}")
                results[test_module] = 'PASSED'
            else:
                print(f"\n⚠ {test_module.replace('_', ' ').title()} - class not found")
                results[test_module] = 'SKIPPED'
        
        except Exception as e:
            print(f"\n✗ {test_module.replace('_', ' ').title()} - {str(e)[:80]}")
            results[test_module] = 'FAILED'
    
    print("\n" + "-" * 80)
    print("[2/3] Generating visualizations...")
    print("-" * 80)
    
    try:
        from visualization_reporter import VisualizationReporter
        reporter = VisualizationReporter()
        reporter.generate_all()
        print("\n✓ Visualizations generated successfully")
    except Exception as e:
        print(f"\n⚠ Visualization generation failed: {str(e)}")
    
    print("\n" + "-" * 80)
    print("[3/3] Summary & Output Locations")
    print("-" * 80)
    
    results_dir = protocol_dir / 'results'
    plots_dir = protocol_dir / 'plots'
    reports_dir = protocol_dir / 'reports'
    
    print(f"\n📊 Results Directory: {results_dir}")
    if results_dir.exists():
        result_files = list(results_dir.glob('*.json'))
        print(f"   Files: {len(result_files)} JSON result files")
        for rf in sorted(result_files)[:5]:
            print(f"   - {rf.name}")
        if len(result_files) > 5:
            print(f"   ... and {len(result_files)-5} more")
    
    print(f"\n📈 Plots Directory: {plots_dir}")
    if plots_dir.exists():
        plot_files = list(plots_dir.glob('*.png'))
        print(f"   Files: {len(plot_files)} PNG plot files")
        for pf in sorted(plot_files):
            print(f"   - {pf.name}")
    
    print(f"\n📄 Reports Directory: {reports_dir}")
    if reports_dir.exists():
        report_files = list(reports_dir.glob('*.*'))
        print(f"   Files: {len(report_files)} report files")
        for rf in sorted(report_files):
            print(f"   - {rf.name}")
    
    print("\n" + "="*80)
    print(" "*15 + "PROTOCOL V2 QUICK START COMPLETE")
    print("="*80)
    
    print("\n📋 Test Results Summary:")
    for test, status in results.items():
        symbol = "✓" if status == "PASSED" else "⚠" if status == "SKIPPED" else "✗"
        print(f"   {symbol} {test.replace('_', ' ').title()}: {status}")
    
    print("\n📚 Next Steps:")
    print("   1. Review JSON results in: " + str(results_dir))
    print("   2. View plots in: " + str(plots_dir))
    print("   3. Read summary report: " + str(reports_dir / 'summary_report.md'))
    print("   4. Check protocol_v2_summary.json for overall metrics")
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    os.chdir(Path(__file__).parent)
    quick_start()
