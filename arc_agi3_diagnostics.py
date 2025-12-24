"""
ARC-AGI-3 Agent Diagnosis & Performance Analysis
Identifies why agents underperform and suggests improvements
"""

import json
import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# ============================================================================
# Performance Diagnostics
# ============================================================================

class AgentDiagnostics:
    """Analyzes agent performance and identifies issues"""
    
    def __init__(self, results_file: str):
        self.results_file = Path(results_file)
        self.results = self._load_results()
        self.diagnostics = {}
    
    def _load_results(self) -> Dict:
        """Load results from JSON"""
        if not self.results_file.exists():
            logger.warning(f"Results file not found: {self.results_file}")
            return {}
        
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def diagnose_agent(self, agent_name: str) -> Dict[str, Any]:
        """Run full diagnostics on agent"""
        
        if not self.results:
            return self._empty_diagnosis(agent_name)
        
        agent_data = self.results.get('agents', {}).get(agent_name, {})
        
        if not agent_data:
            return self._empty_diagnosis(agent_name)
        
        diagnosis = {
            'agent_name': agent_name,
            'score': agent_data.get('mean_score', 0),
            'target': 0.5,
            'below_target': agent_data.get('mean_score', 0) < 0.5,
            'gap_percentage': (0.5 - agent_data.get('mean_score', 0)) * 100,
            'issues': self._identify_issues(agent_data),
            'recommendations': self._generate_recommendations(agent_data),
            'root_causes': self._analyze_root_causes(agent_data),
            'quick_fixes': self._suggest_quick_fixes(agent_data),
        }
        
        self.diagnostics[agent_name] = diagnosis
        return diagnosis
    
    def _empty_diagnosis(self, agent_name: str) -> Dict:
        return {
            'agent_name': agent_name,
            'score': 0,
            'target': 0.5,
            'below_target': True,
            'gap_percentage': 50.0,
            'issues': [],
            'recommendations': [],
            'root_causes': [],
            'quick_fixes': [],
        }
    
    def _identify_issues(self, agent_data: Dict) -> List[str]:
        """Identify performance issues"""
        issues = []
        
        score = agent_data.get('mean_score', 0)
        win_rate = agent_data.get('win_rate', 0)
        mean_steps = agent_data.get('mean_steps', 100)
        
        # Issue 1: Low overall score
        if score < 0.3:
            issues.append("CRITICAL: Agent scoring below 30% - fundamental strategy issue")
        elif score < 0.5:
            issues.append("Agent below 50% target - needs improvement")
        
        # Issue 2: No wins
        if win_rate == 0:
            issues.append("Agent never wins - unable to satisfy any win conditions")
        elif win_rate < 0.1:
            issues.append("Very low win rate (<10%) - strategy not working")
        
        # Issue 3: Too many steps
        if mean_steps > 80:
            issues.append("Taking too many steps (>80) - inefficient action selection")
        elif mean_steps < 10:
            issues.append("Taking very few steps (<10) - giving up too quickly")
        
        # Issue 4: Variance issues
        std_score = agent_data.get('std_score', 0)
        if std_score > score * 0.5:
            issues.append("High variance in performance - inconsistent strategy")
        
        # Issue 5: Random baseline comparison
        if 'Random Baseline' in [a for a in agent_data.keys()]:
            improvement = (score - 0.5) / 0.5 if score > 0 else -1
            if improvement < 0.1:
                issues.append("Minimal improvement over random (< 10%) - strategy too similar to random")
        
        return issues
    
    def _analyze_root_causes(self, agent_data: Dict) -> List[str]:
        """Identify root causes of underperformance"""
        causes = []
        
        score = agent_data.get('mean_score', 0)
        win_rate = agent_data.get('win_rate', 0)
        
        if score < 0.5:
            # Root cause 1: Action space mismatch
            causes.append({
                'cause': 'Action Space Mismatch',
                'description': 'Agent may not be using effective actions for win conditions',
                'evidence': f'Score {score:.1%} suggests ineffective action selection',
                'severity': 'HIGH'
            })
            
            # Root cause 2: Game understanding
            if win_rate < 0.2:
                causes.append({
                    'cause': 'Game Understanding',
                    'description': 'Agent does not understand win conditions or game mechanics',
                    'evidence': f'Win rate {win_rate:.1%} indicates no valid winning strategy found',
                    'severity': 'CRITICAL'
                })
            
            # Root cause 3: Pattern recognition
            causes.append({
                'cause': 'Pattern Recognition',
                'description': 'Agent fails to recognize relevant patterns in game grids',
                'evidence': f'Below-target score despite pattern analyzer',
                'severity': 'HIGH'
            })
            
            # Root cause 4: Memory/learning
            causes.append({
                'cause': 'Learning from History',
                'description': 'Agent not effectively learning from previous actions',
                'evidence': 'Variance in scores suggests inconsistent strategy',
                'severity': 'MEDIUM'
            })
        
        return causes
    
    def _generate_recommendations(self, agent_data: Dict) -> List[str]:
        """Generate specific recommendations for improvement"""
        recs = []
        
        score = agent_data.get('mean_score', 0)
        win_rate = agent_data.get('win_rate', 0)
        mean_steps = agent_data.get('mean_steps', 100)
        
        # Recommendation 1: Game mechanics understanding
        if win_rate < 0.3:
            recs.append("PRIORITY 1: Understand win conditions - run game state analysis to identify what triggers wins")
        
        # Recommendation 2: Action effectiveness
        recs.append("PRIORITY 2: Analyze which actions lead to grid changes - focus on effective actions")
        
        # Recommendation 3: Step efficiency
        if mean_steps > 70:
            recs.append("PRIORITY 3: Reduce step count - implement early stopping for unproductive sequences")
        
        # Recommendation 4: Reward shaping
        recs.append("PRIORITY 4: Implement proper reward shaping - currently rewards are sparse")
        
        # Recommendation 5: State representation
        recs.append("PRIORITY 5: Improve state representation - add more discriminative features")
        
        # Recommendation 6: Exploration strategy
        recs.append("PRIORITY 6: Tune exploration vs exploitation - consciousness layer not optimized")
        
        return recs
    
    def _suggest_quick_fixes(self, agent_data: Dict) -> List[Dict[str, str]]:
        """Suggest quick code fixes"""
        fixes = []
        
        score = agent_data.get('mean_score', 0)
        
        if score < 0.5:
            fixes.append({
                'fix': 'Implement Action Analysis',
                'code': '''
# Track which actions produce grid changes
action_effectiveness = {}
for action in actions:
    effect = execute_action(action)
    action_effectiveness[action] = effect['grid_change']

# Prefer effective actions
best_actions = sorted(action_effectiveness, 
                     key=lambda a: action_effectiveness[a], 
                     reverse=True)[:3]
return random.choice(best_actions)
                ''',
                'expected_improvement': '+10-15%'
            })
            
            fixes.append({
                'fix': 'Add Win Condition Detection',
                'code': '''
# Detect what actions lead to win
win_history = []
for action in action_history:
    if check_win_after_action(action):
        win_history.append(action)

# Repeat successful actions
if win_history:
    return win_history[-1]
                ''',
                'expected_improvement': '+20-30%'
            })
            
            fixes.append({
                'fix': 'Implement State Value Estimation',
                'code': '''
# Learn value of different grid states
state_values = {}
for grid, reward in history:
    state_hash = hash_grid(grid)
    state_values[state_hash] = reward

# Choose actions that lead to high-value states
next_states = get_next_states(current_grid, actions)
values = [state_values.get(hash_grid(s), 0) for s in next_states]
return actions[np.argmax(values)]
                ''',
                'expected_improvement': '+25-35%'
            })
        
        return fixes
    
    def print_diagnosis(self, agent_name: str):
        """Print diagnostic report"""
        diagnosis = self.diagnostics.get(agent_name)
        
        if not diagnosis:
            diagnosis = self.diagnose_agent(agent_name)
        
        print("\n" + "="*80)
        print(f"DIAGNOSTIC REPORT: {agent_name}")
        print("="*80)
        
        print(f"\nPerformance Score: {diagnosis['score']:.1%}")
        print(f"Target Score:      {diagnosis['target']:.0%}")
        
        if diagnosis['below_target']:
            print(f"STATUS: BELOW TARGET (Gap: {diagnosis['gap_percentage']:.1f}%)")
        else:
            print(f"STATUS: TARGET ACHIEVED")
        
        # Issues
        if diagnosis['issues']:
            print("\n" + "-"*80)
            print("IDENTIFIED ISSUES:")
            print("-"*80)
            for i, issue in enumerate(diagnosis['issues'], 1):
                print(f"{i}. {issue}")
        
        # Root causes
        if diagnosis['root_causes']:
            print("\n" + "-"*80)
            print("ROOT CAUSE ANALYSIS:")
            print("-"*80)
            for cause in diagnosis['root_causes']:
                print(f"\nCause: {cause['cause']} [{cause['severity']}]")
                print(f"Description: {cause['description']}")
                print(f"Evidence: {cause['evidence']}")
        
        # Recommendations
        if diagnosis['recommendations']:
            print("\n" + "-"*80)
            print("RECOMMENDATIONS:")
            print("-"*80)
            for rec in diagnosis['recommendations']:
                print(f"• {rec}")
        
        # Quick fixes
        if diagnosis['quick_fixes']:
            print("\n" + "-"*80)
            print("QUICK FIXES (with expected improvement):")
            print("-"*80)
            for fix in diagnosis['quick_fixes']:
                print(f"\nFix: {fix['fix']}")
                print(f"Expected: {fix['expected_improvement']}")
                print(f"Code:\n{fix['code']}")
    
    def print_all_diagnostics(self):
        """Print diagnostics for all agents"""
        for agent_name in self.results.get('agents', {}).keys():
            self.print_diagnosis(agent_name)

# ============================================================================
# Performance Enhancement Suggestions
# ============================================================================

class EnhancementPlanner:
    """Plans enhancements to improve agent performance"""
    
    def __init__(self, current_score: float, target_score: float = 0.5):
        self.current_score = current_score
        self.target_score = target_score
        self.gap = target_score - current_score
    
    def plan_enhancements(self) -> Dict[str, Any]:
        """Plan step-by-step enhancements"""
        
        plan = {
            'current_score': self.current_score,
            'target_score': self.target_score,
            'gap_to_close': self.gap,
            'phases': []
        }
        
        # Phase 1: Fix fundamental issues
        phase1 = {
            'phase': 1,
            'name': 'Foundation (Expected: +10-15%)',
            'duration': '1 hour',
            'items': [
                'Implement proper action-effect tracking',
                'Add grid state hashing and caching',
                'Implement basic reward shaping',
                'Fix exploration/exploitation balance',
            ]
        }
        plan['phases'].append(phase1)
        
        # Phase 2: Add learning
        phase2 = {
            'phase': 2,
            'name': 'Learning (Expected: +15-25%)',
            'duration': '2 hours',
            'items': [
                'Implement state-value estimation',
                'Add action-reward association learning',
                'Implement experience replay buffer',
                'Add win condition pattern learning',
            ]
        }
        plan['phases'].append(phase2)
        
        # Phase 3: Optimize MirrorMind integration
        phase3 = {
            'phase': 3,
            'name': 'MirrorMind Optimization (Expected: +10-20%)',
            'duration': '2 hours',
            'items': [
                'Properly integrate consciousness layer feedback',
                'Implement memory consolidation for game patterns',
                'Add attention mechanism to important features',
                'Tune learning rate based on uncertainty',
            ]
        }
        plan['phases'].append(phase3)
        
        # Phase 4: Advanced features
        phase4 = {
            'phase': 4,
            'name': 'Advanced (Expected: +5-15%)',
            'duration': '3 hours',
            'items': [
                'Implement multi-game strategy transfer',
                'Add meta-learning for quick adaptation',
                'Implement curiosity-driven exploration',
                'Add adversarial robustness',
            ]
        }
        plan['phases'].append(phase4)
        
        plan['total_expected_improvement'] = sum([
            15, 20, 15, 10  # Max of each phase
        ])
        
        return plan
    
    def print_plan(self):
        """Print enhancement plan"""
        plan = self.plan_enhancements()
        
        print("\n" + "="*80)
        print("PERFORMANCE ENHANCEMENT PLAN")
        print("="*80)
        
        print(f"\nCurrent Score:    {plan['current_score']:.1%}")
        print(f"Target Score:     {plan['target_score']:.0%}")
        print(f"Gap to Close:     {plan['gap_to_close']:.1%}")
        
        total_expected = plan['total_expected_improvement']
        projected = plan['current_score'] + total_expected / 100
        
        print(f"\nTotal Expected Improvement: +{total_expected}%")
        print(f"Projected Final Score: {min(projected, 1.0):.1%}")
        
        print("\n" + "-"*80)
        print("PHASES:")
        print("-"*80)
        
        for phase in plan['phases']:
            print(f"\n{phase['name']}")
            print(f"Duration: {phase['duration']}")
            for item in phase['items']:
                print(f"  • {item}")

if __name__ == "__main__":
    # Example usage
    results_file = "arc_agi_results/comparison_results.json"
    
    diag = AgentDiagnostics(results_file)
    diag.print_all_diagnostics()
    
    # Enhancement plan
    current_score = 0.35  # Example
    planner = EnhancementPlanner(current_score, target_score=0.5)
    planner.print_plan()
