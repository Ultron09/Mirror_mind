#!/usr/bin/env python3
"""
Quick Protocol_v3 test runner - minimal version for debugging
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('QuickTest')

# Import Protocol_v3
try:
    from protocol_v3 import (
        ProtocolV3Orchestrator,
        ContinualLearningTestSuite,
        ConsciousnessTestSuite,
        StabilityTestSuite,
    )
    logger.info("✅ Protocol_v3 imported successfully")
except Exception as e:
    logger.error(f"❌ Failed to import: {e}")
    sys.exit(1)

# Create simple test framework
class SimpleFramework(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def main():
    logger.info("\n" + "="*70)
    logger.info("PROTOCOL_V3: QUICK TEST EXECUTION")
    logger.info("="*70 + "\n")
    
    # Create framework
    logger.info("🔧 Creating test framework...")
    framework = SimpleFramework()
    logger.info("✅ Framework created\n")
    
    # Create orchestrator
    logger.info("📋 Creating test orchestrator...")
    orchestrator = ProtocolV3Orchestrator(output_dir='protocol_v3_quick_results')
    logger.info("✅ Orchestrator created\n")
    
    # Register ONLY quick tests
    logger.info("📊 Registering quick tests...")
    orchestrator.register_test(ContinualLearningTestSuite(num_tasks=3, task_length=20))
    orchestrator.register_test(ConsciousnessTestSuite(num_steps=50))
    orchestrator.register_test(StabilityTestSuite(num_steps=100))
    logger.info("✅ Tests registered\n")
    
    # Run tests
    logger.info("🚀 Running tests...\n")
    try:
        report = orchestrator.run_all_tests(framework)
        
        logger.info("\n" + "="*70)
        logger.info("✅ TESTS COMPLETED SUCCESSFULLY")
        logger.info("="*70 + "\n")
        
        # Print summary
        summary = report.get('summary_statistics', {})
        logger.info("📊 RESULTS SUMMARY:")
        logger.info("-" * 70)
        logger.info(f"  Tests Passed: {summary.get('tests_passed', 0)}")
        logger.info(f"  Tests Failed: {summary.get('tests_failed', 0)}")
        logger.info(f"  Duration: {report.get('duration_seconds', 0):.1f} seconds\n")
        
        if 'continual_learning_accuracy' in summary:
            acc = summary['continual_learning_accuracy']
            logger.info(f"  📈 Continual Learning Accuracy: {acc:.4f}")
            logger.info(f"     Target: 0.92 | MIT Seal: 0.85 | Status: {'✅ BEATS TARGET' if acc > 0.92 else '✓ Good' if acc > 0.75 else '❌ Needs work'}\n")
        
        if 'has_consciousness' in summary:
            has_cs = summary.get('has_consciousness', False)
            logger.info(f"  ✨ Consciousness Layer: {'✅ ENABLED & WORKING' if has_cs else '⚠️ Not available'}\n")
        
        # Save results to file for verification
        results_path = Path('protocol_v3_quick_results') / 'quick_test_results.json'
        with open(results_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"💾 Full results saved to: {results_path}")
        logger.info("="*70 + "\n")
        
        return 0
    
    except Exception as e:
        logger.error(f"\n❌ TEST EXECUTION FAILED: {str(e)}")
        logger.error(f"   Error type: {type(e).__name__}")
        import traceback
        logger.error(f"\nTraceback:\n{traceback.format_exc()}")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
