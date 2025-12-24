"""
ARC-AGI-3 Agent Evaluation Framework
Simulates game environments and measures agent performance
"""

import numpy as np
import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import random
from pathlib import Path
import time

# Import agent
from arc_agi3_agent import (
    ArcAgi3MirrorMindAgent, RandomAgent, HeuristicAgent,
    PatternAnalyzer
)

logger = logging.getLogger(__name__)

# ============================================================================
# Game Simulation
# ============================================================================

@dataclass
class GameResult:
    """Result from a single game"""
    game_id: str
    agent_name: str
    score: float
    steps_taken: int
    max_steps: int
    win: bool
    actions_taken: List[str]
    grid_changes: int
    timestamp: str

class SimulatedGame:
    """Simulates an ARC-AGI-3 game"""
    
    def __init__(self, game_id: str, difficulty: int = 5):
        self.game_id = game_id
        self.difficulty = difficulty
        self.grid_size = 10 + difficulty
        self.initial_grid = self._generate_initial_grid()
        self.current_grid = self.initial_grid.copy()
        self.max_steps = 100 + difficulty * 10
        self.step_count = 0
        self.win_condition = None
        self.action_history = []
        
        self._setup_win_condition()
    
    def _generate_initial_grid(self) -> np.ndarray:
        """Generate initial game grid"""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # Add some colored regions
        num_regions = random.randint(2, 5)
        for _ in range(num_regions):
            color = random.randint(1, 9)
            size = random.randint(3, 8)
            y = random.randint(0, self.grid_size - size)
            x = random.randint(0, self.grid_size - size)
            grid[y:y+size, x:x+size] = color
        
        return grid
    
    def _setup_win_condition(self):
        """Define win condition for this game"""
        self.win_condition = random.choice([
            'fill_all',           # Fill all empty cells
            'find_center',        # Fill center cell
            'pattern_match',      # Match a specific pattern
            'color_threshold',    # Reach color threshold
        ])
    
    def get_state(self) -> str:
        """Get current game state"""
        if self.step_count >= self.max_steps:
            return 'GAME_OVER'
        if self._check_win():
            return 'WIN'
        return 'IN_PROGRESS'
    
    def _check_win(self) -> bool:
        """Check if win condition is met"""
        if self.win_condition == 'fill_all':
            return np.sum(self.current_grid) > np.sum(self.initial_grid) * 1.5
        elif self.win_condition == 'find_center':
            center = self.grid_size // 2
            return self.current_grid[center, center] > 0
        elif self.win_condition == 'pattern_match':
            return np.sum(self.current_grid) > self.grid_size ** 2 * 0.3
        elif self.win_condition == 'color_threshold':
            return np.unique(self.current_grid).size > 5
        return False
    
    def get_available_actions(self) -> List[str]:
        """Get list of available actions"""
        return ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'INTERACT', 'RESET']
    
    def execute_action(self, action: str) -> Dict:
        """Execute action and return reward"""
        self.step_count += 1
        self.action_history.append(action)
        
        reward = -0.1  # Time penalty
        grid_change = 0
        
        if action == 'RESET':
            self.current_grid = self.initial_grid.copy()
            reward = -1.0
        elif action == 'INTERACT':
            # Random effect
            if random.random() > 0.5:
                x = random.randint(0, self.grid_size - 1)
                y = random.randint(0, self.grid_size - 1)
                old_val = self.current_grid[y, x]
                self.current_grid[y, x] = random.randint(0, 9)
                if self.current_grid[y, x] != old_val:
                    grid_change = 1
                    reward = 0.2  # Reward for change
        elif action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            # Movement effect - shift pattern
            if random.random() > 0.3:  # Sometimes effective
                if action == 'UP':
                    self.current_grid = np.roll(self.current_grid, -1, axis=0)
                elif action == 'DOWN':
                    self.current_grid = np.roll(self.current_grid, 1, axis=0)
                elif action == 'LEFT':
                    self.current_grid = np.roll(self.current_grid, -1, axis=1)
                elif action == 'RIGHT':
                    self.current_grid = np.roll(self.current_grid, 1, axis=1)
                grid_change = 1
        elif action == 'WAIT':
            reward = -0.05
        
        # Bonus for winning
        if self._check_win():
            reward = 1.0
        
        return {
            'reward': reward,
            'grid_change': grid_change,
            'state': self.get_state(),
        }
    
    def reset(self):
        """Reset game"""
        self.current_grid = self.initial_grid.copy()
        self.step_count = 0
        self.action_history = []
    
    def get_score(self) -> float:
        """Calculate game score"""
        win_bonus = 1.0 if self._check_win() else 0.0
        efficiency = 1.0 - (self.step_count / self.max_steps)
        score = win_bonus * 0.5 + efficiency * 0.5
        return max(0.0, min(1.0, score))

# ============================================================================
# Evaluation Framework
# ============================================================================

class AgentEvaluator:
    """Evaluates agents on multiple games"""
    
    def __init__(self, num_games: int = 10, output_dir: str = "arc_agi_results"):
        self.num_games = num_games
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = []
        self.agent_stats = {}
    
    def create_test_games(self) -> List[SimulatedGame]:
        """Create test games with varying difficulty"""
        games = []
        for i in range(self.num_games):
            difficulty = (i % 5) + 1  # Difficulties 1-5
            game = SimulatedGame(f"game_{i:03d}", difficulty=difficulty)
            games.append(game)
        return games
    
    def evaluate_agent(self, agent: Any, games: List[SimulatedGame], 
                      agent_name: Optional[str] = None) -> Dict:
        """Evaluate single agent on all games"""
        
        if agent_name is None:
            agent_name = agent.name if hasattr(agent, 'name') else "Unknown"
        
        logger.info(f"Evaluating {agent_name} on {len(games)} games...")
        
        game_results = []
        agent_scores = []
        
        for game in games:
            game.reset()
            if hasattr(agent, 'reset'):
                agent.reset()
            
            result = self._run_single_game(agent, game, agent_name)
            game_results.append(result)
            agent_scores.append(result.score)
            
            logger.info(f"  {game.game_id}: score={result.score:.3f}, steps={result.steps_taken}")
        
        # Calculate statistics
        stats = {
            'agent_name': agent_name,
            'num_games': len(games),
            'mean_score': float(np.mean(agent_scores)),
            'std_score': float(np.std(agent_scores)),
            'max_score': float(np.max(agent_scores)),
            'min_score': float(np.min(agent_scores)),
            'win_rate': float(np.mean([r.win for r in game_results])),
            'mean_steps': float(np.mean([r.steps_taken for r in game_results])),
            'games_won': sum([r.win for r in game_results]),
            'evaluation_time': datetime.now().isoformat(),
        }
        
        self.agent_stats[agent_name] = stats
        self.results.extend(game_results)
        
        return stats
    
    def _run_single_game(self, agent: Any, game: SimulatedGame, 
                        agent_name: str) -> GameResult:
        """Run agent on single game"""
        
        actions_taken = []
        grid_changes = 0
        
        while game.get_state() == 'IN_PROGRESS' and game.step_count < game.max_steps:
            try:
                # Get agent decision
                action = agent.decide_action(game.current_grid, game.get_available_actions())
                actions_taken.append(action)
                
                # Execute action
                result = game.execute_action(action)
                grid_changes += result.get('grid_change', 0)
                
            except Exception as e:
                logger.warning(f"Agent error in {game.game_id}: {e}")
                actions_taken.append('ERROR')
                break
        
        # Get final score
        score = game.get_score()
        
        return GameResult(
            game_id=game.game_id,
            agent_name=agent_name,
            score=score,
            steps_taken=game.step_count,
            max_steps=game.max_steps,
            win=game._check_win(),
            actions_taken=actions_taken,
            grid_changes=grid_changes,
            timestamp=datetime.now().isoformat()
        )
    
    def compare_agents(self, agents: Dict[str, Any], games: List[SimulatedGame]) -> Dict:
        """Compare multiple agents"""
        
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'num_games': len(games),
            'agents': {},
            'rankings': [],
        }
        
        for agent_name, agent in agents.items():
            logger.info(f"\n{'='*70}")
            logger.info(f"Testing {agent_name}")
            logger.info(f"{'='*70}")
            
            stats = self.evaluate_agent(agent, games, agent_name=agent_name)
            comparison['agents'][agent_name] = stats
        
        # Rank agents
        ranked = sorted(
            comparison['agents'].items(),
            key=lambda x: x[1]['mean_score'],
            reverse=True
        )
        
        for rank, (name, stats) in enumerate(ranked, 1):
            comparison['rankings'].append({
                'rank': rank,
                'agent_name': name,
                'mean_score': stats['mean_score'],
                'win_rate': stats['win_rate'],
                'games_won': stats['games_won'],
            })
        
        return comparison
    
    def save_results(self, comparison: Dict, filename: str = "comparison_results.json"):
        """Save results to JSON"""
        output_file = self.output_dir / filename
        
        # Convert numpy types to native Python types
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        comparison = convert_types(comparison)
        
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"\nResults saved to: {output_file}")
        return output_file
    
    def print_summary(self, comparison: Dict):
        """Print summary of comparison"""
        
        print("\n" + "="*70)
        print("ARC-AGI-3 AGENT COMPARISON RESULTS")
        print("="*70)
        
        print(f"\nTotal Games Evaluated: {comparison['num_games']}")
        print(f"Evaluation Time: {comparison['timestamp']}\n")
        
        # Rankings
        print("RANKINGS:")
        print("-" * 70)
        print(f"{'Rank':<6} {'Agent':<25} {'Mean Score':<15} {'Win Rate':<15}")
        print("-" * 70)
        
        for ranking in comparison['rankings']:
            print(f"{ranking['rank']:<6} {ranking['agent_name']:<25} "
                  f"{ranking['mean_score']:.3f}       {ranking['win_rate']:.1%}")
        
        print("\n" + "="*70)
        print("DETAILED STATISTICS:")
        print("="*70 + "\n")
        
        for agent_name, stats in comparison['agents'].items():
            print(f"{agent_name}:")
            print(f"  Mean Score:    {stats['mean_score']:.3f} ± {stats['std_score']:.3f}")
            print(f"  Score Range:   {stats['min_score']:.3f} - {stats['max_score']:.3f}")
            print(f"  Win Rate:      {stats['win_rate']:.1%} ({stats['games_won']}/{stats['num_games']} games)")
            print(f"  Mean Steps:    {stats['mean_steps']:.1f}")
            print()

# ============================================================================
# Main Evaluation Script
# ============================================================================

def main():
    """Run full evaluation"""
    
    # Create evaluator
    evaluator = AgentEvaluator(num_games=15, output_dir="arc_agi_results")
    
    # Create test games
    games = evaluator.create_test_games()
    logger.info(f"Created {len(games)} test games")
    
    # Create agents
    agents = {
        'MirrorMind-AGI3': ArcAgi3MirrorMindAgent(use_mirrormimd=True),
        'Random Baseline': RandomAgent(),
        'Heuristic Baseline': HeuristicAgent(),
    }
    
    # Run comparison
    comparison = evaluator.compare_agents(agents, games)
    
    # Save and display results
    evaluator.save_results(comparison)
    evaluator.print_summary(comparison)
    
    # Print detailed metrics
    print("\n" + "="*70)
    print("PERFORMANCE ANALYSIS")
    print("="*70 + "\n")
    
    mirrormimd_score = comparison['agents'].get('MirrorMind-AGI3', {}).get('mean_score', 0)
    random_score = comparison['agents'].get('Random Baseline', {}).get('mean_score', 0)
    
    print(f"MirrorMind Mean Score:  {mirrormimd_score:.1%}")
    print(f"Target (50%):           50.0%")
    print(f"Achievement:            {'✅ ACHIEVED' if mirrormimd_score >= 0.5 else '❌ BELOW TARGET'}")
    
    if mirrormimd_score < 0.5:
        print(f"\nGap to target: {(0.5 - mirrormimd_score)*100:.1f}%")
    
    if random_score > 0:
        improvement = (mirrormimd_score - random_score) / random_score * 100
        print(f"Improvement over random: {improvement:+.1f}%")
    
    return comparison

if __name__ == "__main__":
    import logging.config
    
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
        },
        'loggers': {
            '': {
                'handlers': ['default'],
                'level': 'INFO',
                'propagate': True
            }
        }
    })
    
    comparison = main()
