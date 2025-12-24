"""
Complete ARC-AGI-3 Evaluation with All Agents
Compares MirrorMind vs Baselines vs Simplified vs Hybrid
"""

import sys
import logging
import json
from pathlib import Path
from datetime import datetime
import warnings
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('airbornehrs').setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Minimal format to avoid Unicode issues
)
logger = logging.getLogger(__name__)

def main():
    """Complete evaluation pipeline"""
    
    print("\n" + "="*80)
    print("ARC-AGI-3 COMPREHENSIVE AGENT EVALUATION".center(80))
    print("="*80 + "\n")
    
    # Import evaluator
    print("[1/6] Importing components...")
    try:
        from arc_agi3_evaluator import AgentEvaluator
        from arc_agi3_agent import ArcAgi3MirrorMindAgent, RandomAgent, HeuristicAgent
        from arc_agi3_simplified_agent import SimplifiedARCAgent, HybridSmartAgent
        print("      SUCCESS: All components imported\n")
    except Exception as e:
        print(f"      ERROR: {e}\n")
        return False
    
    # Setup environment
    print("[2/6] Setting up test environment...")
    try:
        output_dir = "arc_agi_results"
        Path(output_dir).mkdir(exist_ok=True)
        
        evaluator = AgentEvaluator(num_games=25, output_dir=output_dir)
        games = evaluator.create_test_games()
        print(f"      Created {len(games)} test games\n")
    except Exception as e:
        print(f"      ERROR: {e}\n")
        return False
    
    # Create agents
    print("[3/6] Initializing agents...")
    try:
        agents = {
            'Random': RandomAgent(),
            'Heuristic': HeuristicAgent(),
            'Simplified-Smart': SimplifiedARCAgent(),
            'Hybrid-Smart': HybridSmartAgent(),
            'MirrorMind': ArcAgi3MirrorMindAgent(use_mirrormimd=True),
        }
        print(f"      Created {len(agents)} agents:")
        for name in agents.keys():
            print(f"        - {name}")
        print()
    except Exception as e:
        print(f"      WARNING: {e}")
        print("      Continuing with available agents\n")
        agents = {
            'Random': RandomAgent(),
            'Heuristic': HeuristicAgent(),
            'Simplified-Smart': SimplifiedARCAgent(),
            'Hybrid-Smart': HybridSmartAgent(),
        }
    
    # Run evaluation
    print("[4/6] Running evaluation...")
    print("      (This may take 1-2 minutes)\n")
    try:
        comparison = evaluator.compare_agents(agents, games)
        print("      COMPLETE\n")
    except Exception as e:
        print(f"      ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    
    # Save results
    print("[5/6] Saving results...")
    try:
        results_file = evaluator.save_results(comparison)
        print(f"      Saved to: {results_file}\n")
    except Exception as e:
        print(f"      WARNING: Could not save to JSON: {e}")
        print("      But continuing with analysis\n")
        results_file = None
    
    # Print results
    print("[6/6] Results Summary\n")
    evaluator.print_summary(comparison)
    
    # Detailed analysis
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS".center(80))
    print("="*80 + "\n")
    
    # Get scores
    agents_data = comparison['agents']
    
    print("Agent Performance Summary:")
    print("-" * 80)
    print(f"{'Agent':<30} {'Score':<12} {'Win Rate':<12} {'Improvement':<15}")
    print("-" * 80)
    
    random_score = agents_data.get('Random', {}).get('mean_score', 0)
    
    for agent_name in sorted(agents_data.keys()):
        stats = agents_data[agent_name]
        score = stats.get('mean_score', 0)
        win_rate = stats.get('win_rate', 0)
        
        if agent_name != 'Random':
            improvement = ((score - random_score) / (random_score + 1e-6)) * 100
        else:
            improvement = 0
        
        marker = ""
        if agent_name == 'MirrorMind':
            marker = " <-- PRIMARY"
        
        print(f"{agent_name:<30} {score:>10.1%}  {win_rate:>10.1%}  {improvement:>12.1f}%{marker}")
    
    print("\n" + "="*80)
    print("TARGET ASSESSMENT".center(80))
    print("="*80 + "\n")
    
    mm_stats = agents_data.get('MirrorMind', {})
    mm_score = mm_stats.get('mean_score', 0)
    target = 0.5
    
    print(f"MirrorMind Score:  {mm_score:.1%}")
    print(f"Target Score:      {target:.0%}")
    print(f"Gap:               {(target - mm_score)*100:.1f}%")
    
    if mm_score >= target:
        print(f"\n[SUCCESS] TARGET ACHIEVED!")
        print(f"   MirrorMind scored {mm_score:.1%}, exceeding the 50% target")
    else:
        print(f"\n[BELOW TARGET]")
        print(f"   Need {(target - mm_score)*100:.1f}% more to reach 50%")
    
    # Suggestions
    print("\n" + "="*80)
    print("NEXT STEPS".center(80))
    print("="*80 + "\n")
    
    if mm_score < 0.5:
        print("To improve MirrorMind to 50%+:\n")
        print("1. Implement action value learning")
        print("   - Track which actions produce grid changes")
        print("   - Prefer high-value actions\n")
        print("2. Add win condition detection")
        print("   - Analyze what triggers wins")
        print("   - Repeat successful action sequences\n")
        print("3. Improve state representation")
        print("   - Add more discriminative features")
        print("   - Better entropy/pattern detection\n")
        print("4. Optimize consciousness layer")
        print("   - Fix plasticity_gate error")
        print("   - Properly integrate with decision making\n")
        print("5. Test against official ARC-AGI-3")
        print("   - Use real game API when available")
        print("   - Get concrete feedback\n")
    else:
        print("Congratulations! MirrorMind has achieved the 50%+ target.\n")
        print("To optimize further:\n")
        print("1. Run more comprehensive tests")
        print("2. Conduct ablation studies")
        print("3. Compare against published benchmarks")
        print("4. Deploy to official ARC-AGI-3 platform\n")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
