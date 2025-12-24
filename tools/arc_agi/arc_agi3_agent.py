"""
ARC-AGI-3 Agent powered by MirrorMind Framework
Integrates consciousness-driven reasoning with adaptive learning for game solving
"""

import numpy as np
import torch
import torch.nn as nn
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from enum import Enum
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import MirrorMind if available
try:
    from airbornehrs import AdaptiveFramework, AdaptiveFrameworkConfig
    MIRRORMIMD_AVAILABLE = True
except ImportError:
    MIRRORMIMD_AVAILABLE = False
    logger.warning("MirrorMind not available - using fallback agent")

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class GameFrame:
    """Represents a single game frame/state"""
    grid: np.ndarray  # Current game grid
    observation: np.ndarray  # Normalized observation
    frame_index: int
    state: str  # 'NOT_PLAYED', 'IN_PROGRESS', 'WIN', 'GAME_OVER'
    rewards: Dict[str, float]
    
@dataclass
class GameMemory:
    """Stores game history for learning"""
    frames: List[GameFrame]
    actions: List['GameAction']
    rewards: List[float]
    patterns: Dict[str, Any]
    
class GameAction(Enum):
    """Available game actions"""
    RESET = "RESET"
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    INTERACT = "INTERACT"
    WAIT = "WAIT"
    
    @classmethod
    def movement_actions(cls):
        return [cls.UP, cls.DOWN, cls.LEFT, cls.RIGHT]
    
    @classmethod
    def all_simple_actions(cls):
        return [cls.RESET, cls.UP, cls.DOWN, cls.LEFT, cls.RIGHT, cls.INTERACT, cls.WAIT]

# ============================================================================
# Pattern Analysis Module
# ============================================================================

class PatternAnalyzer:
    """Analyzes game grids to extract meaningful patterns"""
    
    def __init__(self):
        self.pattern_history = deque(maxlen=100)
        self.color_patterns = {}
        self.spatial_patterns = {}
        
    def analyze_frame(self, grid: np.ndarray) -> Dict[str, Any]:
        """Extract features from game grid"""
        if grid is None or grid.size == 0:
            return self._empty_pattern()
            
        grid = np.array(grid)
        
        features = {
            'grid_shape': grid.shape,
            'unique_colors': len(np.unique(grid)),
            'color_distribution': self._get_color_dist(grid),
            'center_of_mass': self._get_center_of_mass(grid),
            'edges_filled': self._check_edges(grid),
            'symmetry': self._check_symmetry(grid),
            'patterns': self._detect_patterns(grid),
            'entropy': self._calculate_entropy(grid),
        }
        
        self.pattern_history.append(features)
        return features
    
    def _empty_pattern(self) -> Dict:
        return {
            'grid_shape': (0, 0),
            'unique_colors': 0,
            'color_distribution': {},
            'center_of_mass': (0, 0),
            'edges_filled': False,
            'symmetry': 0.0,
            'patterns': [],
            'entropy': 0.0,
        }
    
    def _get_color_dist(self, grid: np.ndarray) -> Dict:
        unique, counts = np.unique(grid, return_counts=True)
        return {int(c): int(cnt) for c, cnt in zip(unique, counts)}
    
    def _get_center_of_mass(self, grid: np.ndarray) -> Tuple[float, float]:
        if grid.size == 0:
            return (0.0, 0.0)
        nonzero = np.argwhere(grid != 0)
        if len(nonzero) == 0:
            return (0.0, 0.0)
        return tuple(nonzero.mean(axis=0))
    
    def _check_edges(self, grid: np.ndarray) -> bool:
        if grid.size == 0:
            return False
        edges = np.concatenate([grid[0], grid[-1], grid[:, 0], grid[:, -1]])
        return np.sum(edges) > grid.size * 0.3
    
    def _check_symmetry(self, grid: np.ndarray) -> float:
        if grid.size < 4:
            return 0.0
        h_sym = np.allclose(grid, np.fliplr(grid))
        v_sym = np.allclose(grid, np.flipud(grid))
        return 1.0 if (h_sym or v_sym) else 0.0
    
    def _detect_patterns(self, grid: np.ndarray) -> List[str]:
        patterns = []
        if grid.size < 2:
            return patterns
        
        # Detect continuous regions
        for color in np.unique(grid):
            mask = (grid == color)
            if mask.sum() > 1:
                patterns.append(f"region_color_{int(color)}")
        
        return patterns
    
    def _calculate_entropy(self, grid: np.ndarray) -> float:
        if grid.size == 0:
            return 0.0
        unique, counts = np.unique(grid, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-10))

# ============================================================================
# MirrorMind Decision Engine
# ============================================================================

class MirrorMindDecisionEngine:
    """Decision making using MirrorMind framework"""
    
    def __init__(self, use_consciousness: bool = True):
        self.use_consciousness = use_consciousness and MIRRORMIMD_AVAILABLE
        
        if self.use_consciousness:
            try:
                config = AdaptiveFrameworkConfig(
                    enable_consciousness=True,
                    memory_type='hybrid',
                    device='cpu'
                )
                base_model = nn.Sequential(
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                )
                self.mm = AdaptiveFramework(base_model, config)
                logger.info("MirrorMind consciousness engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize MirrorMind: {e}")
                self.use_consciousness = False
        
        self.decision_history = deque(maxlen=50)
        self.value_estimates = {}
        
    def decide_action(self, patterns: Dict, grid: np.ndarray, 
                     available_actions: List[GameAction],
                     game_memory: GameMemory) -> GameAction:
        """Make decision based on patterns and consciousness"""
        
        # Convert patterns to tensor
        pattern_vector = self._patterns_to_tensor(patterns)
        
        if self.use_consciousness:
            try:
                # Use MirrorMind consciousness for decision making
                metrics = self.mm.train_step(pattern_vector.unsqueeze(0), 
                                            torch.zeros(1, 128),
                                            enable_dream=True)
                
                # Extract consciousness metrics
                consciousness_score = metrics.get('loss', 0.5)
                attention_weights = metrics.get('attention', None)
                
                # High consciousness → explore, low consciousness → exploit
                exploration_prob = min(consciousness_score, 0.5)
            except Exception as e:
                logger.warning(f"MirrorMind error: {e}")
                exploration_prob = 0.3
        else:
            exploration_prob = 0.3
        
        # Exploration vs exploitation
        if random.random() < exploration_prob:
            action = random.choice(available_actions)
        else:
            # Use value estimates to select best action
            action = self._select_greedy_action(patterns, available_actions, game_memory)
        
        self.decision_history.append({
            'action': action,
            'exploration': exploration_prob,
            'patterns': patterns,
        })
        
        return action
    
    def _patterns_to_tensor(self, patterns: Dict) -> torch.Tensor:
        """Convert pattern dict to tensor"""
        features = [
            patterns.get('entropy', 0.0),
            patterns.get('symmetry', 0.0),
            float(patterns.get('unique_colors', 0)) / 256.0,
        ]
        
        # Color distribution
        color_dist = patterns.get('color_distribution', {})
        for i in range(10):
            features.append(float(color_dist.get(i, 0)) / 256.0)
        
        # Pad to 256 dimensions
        features = features[:256]
        features = features + [0.0] * (256 - len(features))
        
        return torch.tensor(features[:256], dtype=torch.float32)
    
    def _select_greedy_action(self, patterns: Dict, 
                             available_actions: List[GameAction],
                             game_memory: GameMemory) -> GameAction:
        """Select best action based on past experience"""
        
        if not game_memory.frames or len(game_memory.actions) == 0:
            return random.choice(available_actions)
        
        # Simple heuristic: prefer actions that changed the grid significantly
        action_scores = {}
        for action in available_actions:
            action_scores[action] = random.random()  # Baseline
        
        # Prefer INTERACT if we have clear patterns
        if patterns.get('unique_colors', 0) > 2:
            action_scores[GameAction.INTERACT] = action_scores.get(GameAction.INTERACT, 0) + 0.3
        
        # Prefer movement toward center of mass
        com = patterns.get('center_of_mass', (0, 0))
        if com[0] > 0.5:
            action_scores[GameAction.DOWN] = action_scores.get(GameAction.DOWN, 0) + 0.2
        
        # Select action with highest score
        best_action = max([a for a in available_actions if a in action_scores],
                         key=lambda a: action_scores.get(a, 0))
        return best_action

# ============================================================================
# Main ARC-AGI-3 Agent
# ============================================================================

class ArcAgi3MirrorMindAgent:
    """Main agent class for ARC-AGI-3 games"""
    
    def __init__(self, use_mirrormimd: bool = True):
        """Initialize agent with MirrorMind integration"""
        self.name = "MirrorMind-ARC-AGI-3"
        self.use_mirrormimd = use_mirrormimd
        
        # Components
        self.pattern_analyzer = PatternAnalyzer()
        self.decision_engine = MirrorMindDecisionEngine(use_consciousness=use_mirrormimd)
        
        # Game state tracking
        self.game_memory = GameMemory(
            frames=[],
            actions=[],
            rewards=[],
            patterns={}
        )
        self.current_grid = None
        self.last_grid = None
        self.step_count = 0
        self.max_steps = 500
        
        logger.info(f"Initialized {self.name}")
    
    def reset(self):
        """Reset agent for new game"""
        self.game_memory = GameMemory(frames=[], actions=[], rewards=[], patterns={})
        self.current_grid = None
        self.last_grid = None
        self.step_count = 0
    
    def observe(self, grid: np.ndarray, state: str = "IN_PROGRESS") -> Dict:
        """Observe game state and extract features"""
        self.last_grid = self.current_grid
        self.current_grid = grid
        
        # Analyze patterns
        patterns = self.pattern_analyzer.analyze_frame(grid)
        self.game_memory.patterns = patterns
        
        # Detect change
        change_magnitude = self._calculate_change_magnitude(grid)
        
        return {
            'patterns': patterns,
            'change': change_magnitude,
            'grid_shape': grid.shape if grid is not None else (0, 0),
            'step': self.step_count,
        }
    
    def decide_action(self, grid: np.ndarray, 
                     available_actions: Optional[List[str]] = None) -> str:
        """Decide next action"""
        
        self.step_count += 1
        
        # Observe state
        observation = self.observe(grid)
        
        # Map action names to enum
        if available_actions is None:
            available_actions = [a.value for a in GameAction.all_simple_actions()]
        
        enum_actions = [GameAction[a] if a != "INTERACT" else GameAction.INTERACT 
                       for a in available_actions 
                       if a in [e.value for e in GameAction]]
        
        if not enum_actions:
            enum_actions = [GameAction.WAIT]
        
        # Get decision from engine
        action_enum = self.decision_engine.decide_action(
            observation['patterns'],
            grid,
            enum_actions,
            self.game_memory
        )
        
        return action_enum.value
    
    def _calculate_change_magnitude(self, grid: np.ndarray) -> float:
        """Calculate how much the grid changed"""
        if self.last_grid is None or grid is None:
            return 0.0
        
        try:
            if self.last_grid.shape != grid.shape:
                return 1.0
            diff = np.sum(self.last_grid != grid)
            magnitude = diff / (grid.size + 1e-6)
            return float(magnitude)
        except:
            return 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            'name': self.name,
            'uses_mirrormimd': self.use_mirrormimd,
            'total_steps': self.step_count,
            'frames_observed': len(self.game_memory.frames),
            'decisions_made': len(self.decision_engine.decision_history),
        }

# ============================================================================
# Baseline Agents for Comparison
# ============================================================================

class RandomAgent:
    """Baseline: Random action selection"""
    
    def __init__(self):
        self.name = "Random"
        self.step_count = 0
    
    def reset(self):
        self.step_count = 0
    
    def decide_action(self, grid: np.ndarray, available_actions: Optional[List[str]] = None) -> str:
        self.step_count += 1
        if available_actions:
            return random.choice(available_actions)
        return random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'INTERACT'])
    
    def get_stats(self) -> Dict:
        return {'name': self.name, 'steps': self.step_count}

class HeuristicAgent:
    """Baseline: Heuristic-based decision making"""
    
    def __init__(self):
        self.name = "Heuristic"
        self.step_count = 0
        self.last_grid = None
    
    def reset(self):
        self.step_count = 0
        self.last_grid = None
    
    def decide_action(self, grid: np.ndarray, available_actions: Optional[List[str]] = None) -> str:
        self.step_count += 1
        
        if available_actions is None:
            available_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'INTERACT']
        
        # Simple heuristic: alternate between movement and interaction
        if self.step_count % 3 == 0 and 'INTERACT' in available_actions:
            return 'INTERACT'
        
        # Move toward center
        movements = [a for a in available_actions if a in ['UP', 'DOWN', 'LEFT', 'RIGHT']]
        if movements:
            return random.choice(movements)
        
        return 'WAIT'
    
    def get_stats(self) -> Dict:
        return {'name': self.name, 'steps': self.step_count}

# ============================================================================
# Utility Functions
# ============================================================================

def create_agent(agent_type: str = "mirrormimd") -> ArcAgi3MirrorMindAgent:
    """Factory function to create agents"""
    if agent_type.lower() == "mirrormimd":
        return ArcAgi3MirrorMindAgent(use_mirrormimd=True)
    elif agent_type.lower() == "random":
        return RandomAgent()
    elif agent_type.lower() == "heuristic":
        return HeuristicAgent()
    else:
        return ArcAgi3MirrorMindAgent(use_mirrormimd=True)

if __name__ == "__main__":
    # Quick test
    agent = ArcAgi3MirrorMindAgent(use_mirrormimd=True)
    
    # Simulate a game
    test_grid = np.random.randint(0, 10, (10, 10))
    
    agent.reset()
    for _ in range(5):
        action = agent.decide_action(test_grid)
        print(f"Action: {action}")
    
    print("Agent stats:", agent.get_stats())
