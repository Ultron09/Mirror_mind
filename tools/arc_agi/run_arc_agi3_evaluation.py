"""
Main Script: ARC-AGI-3 Agent vs MirrorMind Comparison
Complete pipeline: Generate tests → Run agents → Diagnose → Report
"""

import sys
import logging
import json
from pathlib import Path
from datetime import datetime
import warnings

# Suppress MirrorMind logging issues
warnings.filterwarnings('ignore')
logging.getLogger('airbornehrs').setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def print_header(title: str):
    """Print formatted header"""
    print("\n" + "="*80)
    print(title.center(80))
    print("="*80 + "\n")

def main():
    """Main execution pipeline"""
    
    print_header("ARC-AGI-3 Agent Evaluation with MirrorMind Framework")
    
    # Step 1: Import evaluator
    print("Step 1: Initializing evaluation framework...")
    try:
        from arc_agi3_evaluator import AgentEvaluator
        from arc_agi3_agent import ArcAgi3MirrorMindAgent, RandomAgent, HeuristicAgent
        logger.info("✓ Successfully imported agents and evaluator")
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        logger.error("Make sure arc_agi3_agent.py and arc_agi3_evaluator.py are in current directory")
        return False
    
    # Step 2: Create evaluation setup
    print("Step 2: Setting up test environment...")
    try:
        output_dir = "arc_agi_results"
        Path(output_dir).mkdir(exist_ok=True)
        
        evaluator = AgentEvaluator(num_games=20, output_dir=output_dir)
        games = evaluator.create_test_games()
        logger.info(f"✓ Created {len(games)} test games with varying difficulty")
    except Exception as e:
        logger.error(f"✗ Setup failed: {e}")
        return False
    
    # Step 3: Create agents
    print("Step 3: Initializing agents...")
    try:
        agents = {
            'MirrorMind-AGI3': ArcAgi3MirrorMindAgent(use_mirrormimd=True),
            'Random Baseline': RandomAgent(),
            'Heuristic Baseline': HeuristicAgent(),
        }
        logger.info(f"✓ Initialized {len(agents)} agents")
        for name in agents.keys():
            logger.info(f"  - {name}")
    except Exception as e:
        logger.error(f"✗ Agent initialization failed: {e}")
        return False
    
    # Step 4: Run evaluation
    print("\nStep 4: Running evaluation...")
    try:
        comparison = evaluator.compare_agents(agents, games)
        logger.info("✓ Evaluation complete")
    except Exception as e:
        logger.error(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Save results
    print("\nStep 5: Saving results...")
    try:
        results_file = evaluator.save_results(comparison)
        logger.info(f"✓ Results saved to {results_file}")
    except Exception as e:
        logger.error(f"✗ Failed to save results: {e}")
        return False
    
    # Step 6: Print summary
    print("\nStep 6: Performance Summary...")
    evaluator.print_summary(comparison)
    
    # Step 7: Detailed analysis
    print_header("PERFORMANCE ANALYSIS")
    
    mirrormimd_stats = comparison['agents'].get('MirrorMind-AGI3', {})
    random_stats = comparison['agents'].get('Random Baseline', {})
    
    mm_score = mirrormimd_stats.get('mean_score', 0)
    random_score = random_stats.get('mean_score', 0)
    mm_win_rate = mirrormimd_stats.get('win_rate', 0)
    
    print(f"MirrorMind Mean Score:    {mm_score:.1%}")
    print(f"Random Baseline Score:    {random_score:.1%}")
    print(f"MirrorMind Win Rate:      {mm_win_rate:.1%}")
    print(f"\nTarget Score:             50.0%")
    
    gap = (0.5 - mm_score) * 100
    
    if mm_score >= 0.5:
        print(f"\n✅ TARGET ACHIEVED! Score: {mm_score:.1%} (exceeds 50% target)")
    else:
        print(f"\n❌ BELOW TARGET")
        print(f"   Gap to target: {gap:.1f}%")
        print(f"   Improvement needed: {abs(gap):.1f} percentage points")
    
    if random_score > 0:
        improvement = ((mm_score - random_score) / random_score) * 100
        print(f"\nImprovement vs Random:    {improvement:+.1f}%")
    
    # Step 8: Run diagnostics
    print_header("DIAGNOSTIC ANALYSIS")
    
    try:
        from arc_agi3_diagnostics import AgentDiagnostics, EnhancementPlanner
        
        diag = AgentDiagnostics(str(results_file))
        
        # Diagnose MirrorMind
        mm_diagnosis = diag.diagnose_agent('MirrorMind-AGI3')
        
        print(f"\nAgent: MirrorMind-AGI3")
        print(f"Score: {mm_diagnosis['score']:.1%} (Target: {mm_diagnosis['target']:.0%})")
        
        if mm_diagnosis['issues']:
            print("\nIdentified Issues:")
            for issue in mm_diagnosis['issues']:
                print(f"  • {issue}")
        
        if mm_diagnosis['root_causes']:
            print("\nRoot Causes:")
            for cause in mm_diagnosis['root_causes']:
                print(f"  • {cause['cause']} [{cause['severity']}]")
                print(f"    {cause['description']}")
        
        # Show enhancement plan
        if mm_diagnosis['below_target']:
            print("\n" + "-"*80)
            print("ENHANCEMENT PLAN:")
            print("-"*80)
            
            planner = EnhancementPlanner(mm_score, target_score=0.5)
            plan = planner.plan_enhancements()
            
            print(f"\nTotal Expected Improvement: +{plan['total_expected_improvement']}%")
            print(f"Projected Final Score: {min(mm_score + plan['total_expected_improvement']/100, 1.0):.1%}")
            
            print("\nPhases:")
            for phase in plan['phases']:
                print(f"\n  {phase['name']}")
                print(f"  Duration: {phase['duration']}")
                for item in phase['items']:
                    print(f"    • {item}")
        
        # Quick fixes
        if mm_diagnosis['quick_fixes']:
            print("\n" + "-"*80)
            print("QUICK FIXES:")
            print("-"*80)
            
            for i, fix in enumerate(mm_diagnosis['quick_fixes'], 1):
                print(f"\nFix {i}: {fix['fix']}")
                print(f"Expected Improvement: {fix['expected_improvement']}")
    
    except Exception as e:
        logger.warning(f"Could not run full diagnostics: {e}")
    
    # Final summary
    print_header("SUMMARY & NEXT STEPS")
    
    print("Evaluation Complete!")
    print(f"Results saved to: {output_dir}/")
    print(f"\nKey Files:")
    print(f"  • comparison_results.json - Full results data")
    print(f"  • This analysis - Performance breakdown")
    
    if mm_score < 0.5:
        print(f"\nNext Steps to Reach 50%+ Target:")
        print(f"  1. Review the diagnostic analysis above")
        print(f"  2. Implement quick fixes in order of expected improvement")
        print(f"  3. Rerun evaluation to measure improvement")
        print(f"  4. Continue with enhancement phases")
    else:
        print(f"\n✅ MirrorMind has achieved {mm_score:.1%} - TARGET EXCEEDED!")
        print(f"  Continue optimization for even higher scores")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
