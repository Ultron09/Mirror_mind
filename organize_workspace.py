#!/usr/bin/env python3
"""
Workspace Organization Script
Moves files to proper directory structure
"""

import os
import shutil
from pathlib import Path

root = Path("c:/Users/surya/In Use/Personal/UltOrg/Airborne.HRS/MirrorMind")

# Define file movements
movements = {
    # Documentation - Assessment Files
    "docs/assessment": [
        "AIRBORNEHRS_ASSESSMENT.md",
        "AIRBORNEHRS_QUICK_REFERENCE.md",
        "AIRBORNEHRS_EXECUTIVE_SUMMARY.md",
        "HONEST_ASSESSMENT.md",
        "FRAMEWORK_USEFULNESS_ASSESSMENT.md",
    ],
    
    # Documentation - Guides
    "docs/guides": [
        "API.md",
        "GETTING_STARTED.md",
        "FRAMEWORK_README.md",
        "IMPLEMENTATION_GUIDE.md",
        "QUICK_REFERENCE.md",
        "ARCHITECTURE_DETAILS.md",
        "PRESETS_QUICK_START.py",
        "PRESETS_VISUAL_GUIDE.md",
        "PRESETS_INDEX.md",
        "PRESETS_DELIVERY_SUMMARY.md",
        "PRESETS_IMPLEMENTATION_SUMMARY.md",
        "CONSCIOUSNESS_QUICK_START.md",
        "CONSCIOUSNESS_INTEGRATION_COMPLETE.md",
        "SELF_AWARENESS_DOCS.md",
        "SELF_AWARENESS_IMPLEMENTATION.md",
    ],
    
    # Documentation - Reports
    "docs/reports": [
        "MIRRORMING_INTEGRATION_REPORT.md",
        "RESULTS_SUMMARY.md",
        "FINAL_RESULTS.md",
        "FINAL_REPORT.txt",
        "DELIVERY_SUMMARY.md",
        "SESSION_SUMMARY.md",
        "EVERYTHING_COMPLETE.md",
        "IMPLEMENTATION_COMPLETE.md",
        "INTEGRATION_STATUS.md",
        "INTEGRATION_COMPLETE.md",
        "PROTOCOL_V2_COMPLETE.md",
        "PROTOCOL_V3_RESULTS.md",
        "PROTOCOL_V3_SUMMARY.md",
        "PROTOCOL_V3_GUIDE.md",
        "README_PROTOCOL_V3.md",
        "PROTOCOL_V3_SPECIFICATION.md",
        "PROTOCOL_V3_FINAL_INSTRUCTIONS.md",
        "phase8_streaming_report.md",
        "PROTOCOL_V4_UX_ANALYSIS.md",
        "PROTOCOL_V4_USABILITY_TEST.md",
        "Results.md",
        "BUGS_FIXED_SUMMARY.md",
        "BUG_REPORT_AND_FIXES.md",
        "REMAINING_FIXES.md",
        "SOTA_ENHANCEMENT_STRATEGY.md",
        "IMPLEMENTATION_CHECKLIST.md",
    ],
    
    # Tests - Validation
    "tests/validation": [
        "final_verification.py",
        "validate_bug_fixes.py",
        "validate_bug_fixes_clean.py",
        "validate_consciousness.py",
        "test_consciousness.py",
        "test_integration.py",
        "test_training_consciousness.py",
        "test_train_clean.py",
        "ground_truth_verification.py",
    ],
    
    # Tests - Benchmarks
    "tests/benchmarks": [
        "mirrorming_benchmark.py",
        "mirrorming_quick_benchmark.py",
        "airbornehrs_real_world_test.py",
        "airbornehrs_comprehensive_assessment.py",
        "protocol_v3_benchmarks.py",
        "protocol_v3_comprehensive_test.py",
        "quick_protocol_v3_test.py",
    ],
    
    # Results - Benchmarks
    "results/benchmarks": [
        "mirrorming_quick_benchmark_results.json",
        "verification_results.json",
        "GROUND_TRUTH_VERIFICATION.json",
        "robustness_results.json",
        "sweep_results.json",
        "submission.json",
        "evaluation_v2.log",
        "PROTOCOL_V3_FINAL_RESULTS.txt",
        "arc_evaluation_output.txt",
    ],
    
    # Results - Assessments
    "results/assessments": [
        "airbornehrs_comprehensive_assessment.json",
        "airbornehrs_real_world_test_results.json",
        "FRAMEWORK_UX_RESULTS.json",
        "protocol_v4_report.json",
        "protocol_v4_virtual_employee.json",
        "protocol_v4_pathfinder.json",
        "protocol_v4_llm.json",
        "protocol_v4_robot.json",
    ],
    
    # Protocols - V3
    "protocols/v3": [
        "protocol_v3.py",
        "run_protocol_v3.py",
    ],
    
    # Protocols - V4
    "protocols/v4": [
        "protocol_v4.py",
        "protocol_v4_report.md",
    ],
    
    # Tools - ARC AGI
    "tools/arc_agi": [
        "arc_data.py",
        "arc_agi3_agent.py",
        "arc_agi3_agent_v2.py",
        "arc_agi3_diagnostics.py",
        "arc_agi3_evaluator.py",
        "arc_agi3_evaluator_v2.py",
        "arc_agi3_simplified_agent.py",
        "run_arc_agi3_evaluation.py",
        "run_complete_arc_evaluation.py",
    ],
}

# Execute moves
total = 0
moved = 0
errors = []

for dest_dir, files in movements.items():
    dest_path = root / dest_dir
    dest_path.mkdir(parents=True, exist_ok=True)
    
    for file in files:
        source = root / file
        target = dest_path / file
        total += 1
        
        if source.exists() and not target.exists():
            try:
                shutil.move(str(source), str(target))
                moved += 1
                print(f"✅ Moved: {file} → {dest_dir}/")
            except Exception as e:
                errors.append(f"❌ {file}: {e}")
                print(f"❌ Failed: {file}")
        elif source.exists() and target.exists():
            print(f"⚠️  Skipped: {file} (already exists in destination)")
        else:
            print(f"⚠️  Skipped: {file} (not found)")

print(f"\n{'='*80}")
print(f"ORGANIZATION COMPLETE")
print(f"{'='*80}")
print(f"Total files: {total}")
print(f"Moved: {moved}")
print(f"Skipped/Errors: {total - moved}")

if errors:
    print(f"\nErrors:")
    for error in errors:
        print(f"  {error}")
