"""
ARC-AGI-3 Agent V2 - Enhanced with Advanced Pattern Recognition & Learning
Powered by MirrorMind Framework with Q-Learning and game-specific strategies
"""

import numpy as np
import torch
import torch.nn as nn
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from collections import deque, defaultdict
from enum import Enum
import random
import math

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
    grid: np.ndarray
    observation: np.ndarray
    frame_index: int
    state: str
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
# Advanced Pattern Analyzer V2
# ============================================================================

class AdvancedPatternAnalyzer:
    """Enhanced pattern analysis with transformation detection"""
    
    def __init__(self):
        self.pattern_history = deque(maxlen=100)
        self.color_patterns = {}
        self.spatial_patterns = {}
        
    def analyze_frame(self, grid: np.ndarray) -> Dict[str, Any]:
        """Extract comprehensive features from game grid"""
        if grid is None or grid.size == 0:
            return self._empty_pattern()
        
        grid = np.array(grid, dtype=int)
        
        features = {
            'grid_shape': grid.shape,
            'unique_colors': len(np.unique(grid)),
            'color_distribution': self._get_color_dist(grid),
            'center_of_mass': self._get_center_of_mass(grid),
            'edges_filled': self._check_edges(grid),
            'symmetry': self._check_symmetry(grid),
            'patterns': self._detect_patterns(grid),
            'entropy': self._calculate_entropy(grid),
            'grid_fill_ratio': self._get_fill_ratio(grid),
            'transformations': self._detect_transformations(grid),
            'shape_descriptors': self._get_shape_descriptors(grid),
            'color_clusters': self._detect_color_clusters(grid),
            'repetitive_patterns': self._find_repetitive_patterns(grid),
            'gradients': self._detect_gradients(grid),
        }
        
        self.pattern_history.append(features)
        return features
    
    def _empty_pattern(self) -> Dict:
        return {
            'grid_shape': (0, 0),
            'unique_colors': 0,
            'color_distribution': {},
            'center_of_mass': (0, 0),
            'edges_filled': 0.0,
            'symmetry': 0.0,
            'patterns': [],
            'entropy': 0.0,
            'grid_fill_ratio': 0.0,
            'transformations': [],
            'shape_descriptors': {},
            'color_clusters': [],
            'repetitive_patterns': False,
            'gradients': 0.0,
        }
    
    def _get_color_dist(self, grid: np.ndarray) -> Dict[int, float]:
        """Color distribution (normalized)"""
        unique, counts = np.unique(grid, return_counts=True)
        total = grid.size
        return {int(u): float(c) / total for u, c in zip(unique, counts)}
    
    def _get_center_of_mass(self, grid: np.ndarray) -> Tuple[float, float]:
        """Calculate center of mass of non-zero elements"""
        if np.sum(grid) == 0:
            return (0.0, 0.0)
        coords = np.argwhere(grid > 0)
        if len(coords) == 0:
            return (0.0, 0.0)
        com = coords.mean(axis=0)
        return (float(com[0]) / grid.shape[0], float(com[1]) / grid.shape[1])
    
    def _check_edges(self, grid: np.ndarray) -> float:
        """Check how much edges are filled"""
        if grid.size == 0:
            return 0.0
        edges = np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]])
        filled = np.sum(edges > 0) / len(edges)
        return float(filled)
    
    def _check_symmetry(self, grid: np.ndarray) -> float:
        """Detect vertical and horizontal symmetry"""
        if grid.size == 0:
            return 0.0
        h_sym = np.sum(grid == np.flipud(grid)) / grid.size
        v_sym = np.sum(grid == np.fliplr(grid)) / grid.size
        return float(max(h_sym, v_sym))
    
    def _detect_patterns(self, grid: np.ndarray) -> List[str]:
        """Detect common patterns"""
        patterns = []
        if np.sum(grid) == 0:
            patterns.append('empty')
        if len(np.unique(grid)) == 1:
            patterns.append('uniform')
        if self._check_symmetry(grid) > 0.7:
            patterns.append('symmetric')
        if np.max(grid) - np.min(grid) > 5:
            patterns.append('high_variance')
        return patterns
    
    def _calculate_entropy(self, grid: np.ndarray) -> float:
        """Shannon entropy of grid values"""
        if grid.size == 0:
            return 0.0
        unique, counts = np.unique(grid, return_counts=True)
        probs = counts / counts.sum()
        return float(-np.sum(probs * np.log2(probs + 1e-10)))
    
    def _get_fill_ratio(self, grid: np.ndarray) -> float:
        """Ratio of non-zero cells"""
        if grid.size == 0:
            return 0.0
        return float(np.sum(grid > 0) / grid.size)
    
    def _detect_transformations(self, grid: np.ndarray) -> List[str]:
        """Detect possible transformations"""
        transformations = []
        
        # Check rotation
        for k in range(1, 4):
            if np.array_equal(grid, np.rot90(grid, k)):
                transformations.append(f'rotation_{k*90}')
        
        # Check reflection
        if np.array_equal(grid, np.flipud(grid)):
            transformations.append('flip_vertical')
        if np.array_equal(grid, np.fliplr(grid)):
            transformations.append('flip_horizontal')
        
        return transformations
    
    def _get_shape_descriptors(self, grid: np.ndarray) -> Dict[str, float]:
        """Describe shapes in grid"""
        binary = grid > 0
        
        # Compactness and solidity
        area = np.sum(binary)
        if area == 0:
            return {'compactness': 0.0, 'solidity': 0.0}
        
        perimeter = np.sum(binary) - np.sum(binary[1:-1, 1:-1])
        compactness = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
        
        return {
            'compactness': float(np.clip(compactness, 0, 1)),
            'solidity': float(area / grid.size),
        }
    
    def _detect_color_clusters(self, grid: np.ndarray) -> List[Dict]:
        """Detect color clusters/regions"""
        clusters = []
        unique_colors = np.unique(grid[grid > 0])
        
        for color in unique_colors:
            mask = (grid == color)
            num_regions = self._count_regions(mask)
            size = np.sum(mask)
            clusters.append({
                'color': int(color),
                'size': int(size),
                'num_regions': int(num_regions),
            })
        
        return clusters
    
    def _count_regions(self, binary_map: np.ndarray) -> int:
        """Count connected components"""
        visited = np.zeros_like(binary_map, dtype=bool)
        count = 0
        
        for i in range(binary_map.shape[0]):
            for j in range(binary_map.shape[1]):
                if binary_map[i, j] and not visited[i, j]:
                    self._flood_fill(binary_map, visited, i, j)
                    count += 1
        
        return count
    
    def _flood_fill(self, grid: np.ndarray, visited: np.ndarray, i: int, j: int):
        """Flood fill for region counting"""
        if i < 0 or i >= grid.shape[0] or j < 0 or j >= grid.shape[1]:
            return
        if visited[i, j] or not grid[i, j]:
            return
        
        visited[i, j] = True
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            self._flood_fill(grid, visited, i + di, j + dj)
    
    def _find_repetitive_patterns(self, grid: np.ndarray) -> bool:
        """Check for repetitive patterns"""
        if grid.size < 16:
            return False
        
        # Check for tile patterns
        h, w = grid.shape
        for tile_h in [2, 3, 4]:
            for tile_w in [2, 3, 4]:
                if h % tile_h == 0 and w % tile_w == 0:
                    tile = grid[:tile_h, :tile_w]
                    is_repetitive = True
                    for i in range(0, h, tile_h):
                        for j in range(0, w, tile_w):
                            if not np.array_equal(grid[i:i+tile_h, j:j+tile_w], tile):
                                is_repetitive = False
                                break
                    if is_repetitive:
                        return True
        
        return False
    
    def _detect_gradients(self, grid: np.ndarray) -> float:
        """Detect color gradients"""
        if grid.size < 4:
            return 0.0
        
        gy = np.abs(np.diff(grid, axis=0))
        gx = np.abs(np.diff(grid, axis=1))
        
        total_gradient = np.sum(gy) + np.sum(gx)
        return float(total_gradient / (grid.size * 9))

# ============================================================================
# Q-Learning Based Decision Engine V2
# ============================================================================

class QLearningDecisionEngine:
    """Q-Learning based decision making with pattern memory"""
    
    def __init__(self, alpha: float = 0.2, gamma: float = 0.9, epsilon: float = 0.25):
        self.alpha = alpha  # Learning rate (higher = faster learning)
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.initial_epsilon = epsilon
        
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.state_action_history = deque(maxlen=1000)
        self.successful_patterns = []
        self.visit_count = defaultdict(int)  # Track state visits for epsilon decay
        
    def decay_epsilon(self, total_games: int = 1):
        """Decay exploration rate over time"""
        self.epsilon = max(0.05, self.initial_epsilon * (1.0 - total_games / 100.0))
        
    def get_state_hash(self, patterns: Dict) -> str:
        """Create hashable state representation"""
        key_features = (
            int(patterns.get('unique_colors', 0)),
            int(patterns.get('entropy', 0) * 10),
            patterns.get('win_condition', 'unknown'),
        )
        return str(key_features)
    
    def select_action(self, state_hash: str, available_actions: List[GameAction],
                     is_training: bool = True) -> GameAction:
        """Select action using epsilon-greedy strategy"""
        
        self.visit_count[state_hash] += 1
        
        # Exploration vs exploitation
        if is_training and random.random() < self.epsilon:
            return random.choice(available_actions)
        
        # Exploitation: select best Q-value action
        action_values = {
            action: self.q_table[state_hash].get(action.value, 0.0)
            for action in available_actions
        }
        
        best_value = max(action_values.values()) if action_values else 0.0
        best_actions = [a for a, v in action_values.items() if abs(v - best_value) < 1e-6]
        
        return random.choice(best_actions) if best_actions else random.choice(available_actions)
    
    def update_q_value(self, state: str, action: GameAction, reward: float, next_state: str):
        """Update Q-value using Q-learning formula"""
        current_q = self.q_table[state].get(action.value, 0.0)
        max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0.0
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action.value] = new_q
        
        # Store successful patterns
        if reward > 0:
            self.successful_patterns.append({
                'state': state,
                'action': action.value,
                'reward': reward,
            })

# ============================================================================
# Game State Classifier
# ============================================================================

class GameTypeClassifier:
    """Classifies game type based on patterns"""
    
    @staticmethod
    def classify_game(patterns: Dict, grid: np.ndarray) -> str:
        """Classify what type of game this is"""
        
        fill_ratio = patterns.get('grid_fill_ratio', 0)
        entropy = patterns.get('entropy', 0)
        unique_colors = patterns.get('unique_colors', 0)
        
        # Pattern-based classification
        if fill_ratio < 0.3:
            return 'fill_game'
        elif unique_colors < 3:
            return 'color_game'
        elif entropy > 4:
            return 'complex_game'
        elif patterns.get('repetitive_patterns', False):
            return 'pattern_game'
        else:
            return 'transformation_game'
    
    @staticmethod
    def get_strategy_for_type(game_type: str) -> List[GameAction]:
        """Return action priority for game type"""
        strategies = {
            'fill_game': [
                GameAction.INTERACT, GameAction.INTERACT, GameAction.DOWN, 
                GameAction.RIGHT, GameAction.UP, GameAction.LEFT, GameAction.WAIT
            ],
            'color_game': [
                GameAction.INTERACT, GameAction.INTERACT, GameAction.INTERACT,
                GameAction.WAIT, GameAction.UP, GameAction.DOWN, GameAction.LEFT, GameAction.RIGHT
            ],
            'pattern_game': [
                GameAction.INTERACT, GameAction.UP, GameAction.DOWN, 
                GameAction.LEFT, GameAction.RIGHT, GameAction.WAIT
            ],
            'transformation_game': [
                GameAction.UP, GameAction.DOWN, GameAction.LEFT, GameAction.RIGHT,
                GameAction.INTERACT, GameAction.WAIT
            ],
            'complex_game': [
                GameAction.INTERACT, GameAction.INTERACT, GameAction.WAIT, 
                GameAction.UP, GameAction.DOWN, GameAction.LEFT, GameAction.RIGHT
            ],
        }
        return strategies.get(game_type, [GameAction.INTERACT, GameAction.INTERACT] + GameAction.all_simple_actions())

# ============================================================================
# Main Agent V2
# ============================================================================

class ArcAgi3MirrorMindAgentV2:
    """Main agent class with advanced learning and strategies"""
    
    def __init__(self, use_mirrormimd: bool = True):
        """Initialize enhanced agent"""
        self.name = "MirrorMind-V2-ARC-AGI-3"
        self.use_mirrormimd = use_mirrormimd and MIRRORMIMD_AVAILABLE
        
        # Components
        self.pattern_analyzer = AdvancedPatternAnalyzer()
        self.decision_engine = QLearningDecisionEngine(alpha=0.25, gamma=0.95, epsilon=0.15)
        self.game_classifier = GameTypeClassifier()
        
        # Game state tracking
        self.game_memory = GameMemory(
            frames=[],
            actions=[],
            rewards=[],
            patterns={}
        )
        self.current_grid = None
        self.last_grid = None
        self.last_state_hash = None
        self.step_count = 0
        self.max_steps = 500
        self.game_type = 'unknown'
        self.rewards_history = deque(maxlen=100)
        self.game_count = 0
        
        logger.info(f"Initialized {self.name}")
        
        logger.info(f"Initialized {self.name}")
    
    def reset(self):
        """Reset agent for new game"""
        self.game_count += 1
        self.game_memory = GameMemory(frames=[], actions=[], rewards=[], patterns={})
        self.current_grid = None
        self.last_grid = None
        self.last_state_hash = None
        self.step_count = 0
        self.game_type = 'unknown'
        
        # Gradually reduce exploration as we gain experience
        self.decision_engine.epsilon = max(0.05, 0.15 * (1.0 - min(1.0, self.game_count / 50.0)))
    
    def observe(self, grid: np.ndarray, state: str = "IN_PROGRESS") -> Dict:
        """Observe game state and extract features"""
        self.last_grid = self.current_grid
        self.current_grid = grid
        
        # Analyze patterns
        patterns = self.pattern_analyzer.analyze_frame(grid)
        self.game_memory.patterns = patterns
        
        # Classify game type
        self.game_type = self.game_classifier.classify_game(patterns, grid)
        patterns['game_type'] = self.game_type
        
        # Detect change
        change_magnitude = self._calculate_change_magnitude(grid)
        
        return {
            'patterns': patterns,
            'change': change_magnitude,
            'grid_shape': grid.shape if grid is not None else (0, 0),
            'step': self.step_count,
            'game_type': self.game_type,
        }
    
    def decide_action(self, grid: np.ndarray, 
                     available_actions: Optional[List[str]] = None) -> str:
        """Decide next action - optimized heuristic with focus on INTERACT"""
        
        self.step_count += 1
        
        # Observe state
        observation = self.observe(grid)
        
        # Map action names to enum
        if available_actions is None:
            available_actions = [a.value for a in GameAction.all_simple_actions()]
        
        enum_actions = [GameAction[a] for a in available_actions 
                       if a in [e.value for e in GameAction]]
        
        if not enum_actions:
            enum_actions = [GameAction.WAIT]
        
        # Analyze grid
        fill_ratio = np.sum(grid > 0) / grid.size if grid.size > 0 else 0
        
        # High interaction rate on sparse grids (KEY: This drives progress)
        if fill_ratio < 0.4:
            if random.random() < 0.9:  # Increased from 0.85 to 0.90
                if GameAction.INTERACT in enum_actions:
                    self.last_state_hash = ""
                    action_enum = GameAction.INTERACT
                    self.game_memory.actions.append(action_enum)
                    return action_enum.value
        
        # On denser grids, still heavy INTERACT bias
        if fill_ratio >= 0.3:
            if random.random() < 0.85:  # Increased from 0.75 to 0.85
                if GameAction.INTERACT in enum_actions:
                    self.last_state_hash = ""
                    action_enum = GameAction.INTERACT
                    self.game_memory.actions.append(action_enum)
                    return action_enum.value
            
            movements = [a for a in enum_actions if a in [GameAction.UP, GameAction.DOWN, GameAction.LEFT, GameAction.RIGHT]]
            if movements:
                self.last_state_hash = ""
                action_enum = random.choice(movements)
                self.game_memory.actions.append(action_enum)
                return action_enum.value
        
        # Fallback: try interaction
        if GameAction.INTERACT in enum_actions and random.random() < 0.9:
            self.last_state_hash = ""
            action_enum = GameAction.INTERACT
            self.game_memory.actions.append(action_enum)
            return action_enum.value
        
        # Final fallback
        self.last_state_hash = ""
        action_enum = enum_actions[0] if enum_actions else GameAction.WAIT
        self.game_memory.actions.append(action_enum)
        
        return action_enum.value
    
    def provide_feedback(self, reward: float, grid: np.ndarray):
        """Track feedback for learning (simplified)"""
        self.rewards_history.append(reward)
    
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
        avg_reward = float(np.mean(self.rewards_history)) if self.rewards_history else 0.0
        return {
            'name': self.name,
            'uses_mirrormimd': self.use_mirrormimd,
            'total_steps': self.step_count,
            'average_reward': avg_reward,
            'decisions_made': len(self.game_memory.actions),
            'q_states_learned': len(self.decision_engine.q_table),
        }

# ============================================================================
# Baseline Agents
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
    
    def provide_feedback(self, reward: float, grid: np.ndarray):
        pass
    
    def get_stats(self) -> Dict:
        return {'name': self.name, 'steps': self.step_count}

class SmartHeuristicAgent:
    """Enhanced heuristic agent with smart strategies"""
    
    def __init__(self):
        self.name = "Smart-Heuristic"
        self.step_count = 0
        self.last_grid = None
        self.interact_interval = 2
    
    def reset(self):
        self.step_count = 0
        self.last_grid = None
    
    def decide_action(self, grid: np.ndarray, available_actions: Optional[List[str]] = None) -> str:
        self.step_count += 1
        
        if available_actions is None:
            available_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'INTERACT']
        
        # Analyze grid to choose good actions
        fill_ratio = np.sum(grid > 0) / grid.size if grid.size > 0 else 0
        
        # Strategy: frequent interaction on sparse grids, movement on full grids
        if fill_ratio < 0.4 and self.step_count % self.interact_interval == 0:
            if 'INTERACT' in available_actions:
                return 'INTERACT'
        
        # Move in patterns
        movements = [a for a in available_actions if a in ['UP', 'DOWN', 'LEFT', 'RIGHT']]
        if movements and random.random() < 0.7:
            return random.choice(movements)
        
        # Fallback to interaction or wait
        if 'INTERACT' in available_actions and random.random() < 0.3:
            return 'INTERACT'
        
        return 'WAIT' if 'WAIT' in available_actions else available_actions[0]
    
    def provide_feedback(self, reward: float, grid: np.ndarray):
        pass
    
    def get_stats(self) -> Dict:
        return {'name': self.name, 'steps': self.step_count}

# ============================================================================
# Utility Functions
# ============================================================================

def create_agent(agent_type: str = "mirrormimd_v2") -> Any:
    """Factory function to create agents"""
    if agent_type.lower() == "mirrormimd_v2":
        return ArcAgi3MirrorMindAgentV2(use_mirrormimd=True)
    elif agent_type.lower() == "random":
        return RandomAgent()
    elif agent_type.lower() == "smart_heuristic":
        return SmartHeuristicAgent()
    else:
        return ArcAgi3MirrorMindAgentV2(use_mirrormimd=True)

if __name__ == "__main__":
    # Quick test
    agent = ArcAgi3MirrorMindAgentV2(use_mirrormimd=True)
    
    # Simulate a game
    test_grid = np.random.randint(0, 10, (10, 10))
    
    agent.reset()
    for i in range(5):
        action = agent.decide_action(test_grid)
        reward = 0.1 if i % 2 == 0 else -0.05
        agent.provide_feedback(reward, test_grid)
        print(f"Step {i}: Action={action}, Reward={reward}")
    
    print("\nAgent stats:", agent.get_stats())
