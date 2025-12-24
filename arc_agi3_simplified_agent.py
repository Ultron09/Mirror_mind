"""
ARC-AGI-3 Simplified Agent (Without MirrorMind)
Baseline intelligent agent focused on game understanding
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any
from collections import deque
import random
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# Simplified Intelligent Agent
# ============================================================================

class SimplifiedARCAgent:
    """Simplified but intelligent agent for ARC-AGI games"""
    
    def __init__(self):
        self.name = "Simplified-Intelligent"
        self.step_count = 0
        self.grid_history = deque(maxlen=20)
        self.action_history = deque(maxlen=20)
        self.reward_history = deque(maxlen=20)
        self.action_effectiveness = {}  # Track which actions work
        self.state_values = {}  # Learned state values
        
    def reset(self):
        """Reset for new game"""
        self.step_count = 0
        self.grid_history.clear()
        self.action_history.clear()
        self.reward_history.clear()
    
    def observe_reward(self, reward: float, grid: np.ndarray):
        """Learn from feedback"""
        if self.action_history:
            last_action = self.action_history[-1]
            if last_action not in self.action_effectiveness:
                self.action_effectiveness[last_action] = []
            self.action_effectiveness[last_action].append(reward)
        
        self.reward_history.append(reward)
        self.grid_history.append(grid.copy())
    
    def decide_action(self, grid: np.ndarray, available_actions: Optional[List[str]] = None) -> str:
        """Make intelligent decision"""
        
        self.step_count += 1
        
        if available_actions is None:
            available_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'INTERACT']
        
        # Analyze grid
        features = self._analyze_grid(grid)
        
        # Get action scores
        action_scores = self._score_actions(available_actions, features)
        
        # Select action (exploration vs exploitation)
        if random.random() < 0.2:  # 20% exploration
            action = random.choice(available_actions)
        else:  # 80% exploitation
            action = max(action_scores.items(), key=lambda x: x[1])[0]
        
        self.action_history.append(action)
        return action
    
    def _analyze_grid(self, grid: np.ndarray) -> Dict[str, float]:
        """Analyze grid features"""
        features = {
            'unique_colors': float(len(np.unique(grid))),
            'fill_ratio': float(np.sum(grid > 0) / grid.size) if grid.size > 0 else 0,
            'entropy': self._entropy(grid),
            'edge_activity': float(np.sum(grid[[0, -1], :]) + np.sum(grid[:, [0, -1]])),
        }
        return features
    
    def _entropy(self, grid: np.ndarray) -> float:
        """Calculate entropy of grid"""
        if grid.size == 0:
            return 0.0
        unique, counts = np.unique(grid, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    def _score_actions(self, actions: List[str], features: Dict) -> Dict[str, float]:
        """Score available actions"""
        scores = {}
        
        for action in actions:
            base_score = 0.5
            
            # Heuristic scoring
            if action == 'INTERACT':
                base_score += features['unique_colors'] * 0.1
            elif action in ['UP', 'DOWN']:
                base_score += features['fill_ratio'] * 0.2
            elif action in ['LEFT', 'RIGHT']:
                base_score += features['entropy'] * 0.15
            elif action == 'WAIT':
                base_score += 0.1
            
            # Learned effectiveness
            if action in self.action_effectiveness:
                avg_reward = np.mean(self.action_effectiveness[action])
                base_score += avg_reward * 0.3
            
            scores[action] = base_score
        
        return scores
    
    def get_stats(self) -> Dict:
        return {'name': self.name, 'steps': self.step_count}

# ============================================================================
# Hybrid Agent (Mix of strategies)
# ============================================================================

class HybridSmartAgent:
    """Hybrid agent combining multiple strategies"""
    
    def __init__(self):
        self.name = "Hybrid-Smart"
        self.step_count = 0
        self.strategy = "explore"  # explore or exploit
        self.last_score = 0
        self.consecutive_failures = 0
    
    def reset(self):
        self.step_count = 0
        self.strategy = "explore"
        self.last_score = 0
        self.consecutive_failures = 0
    
    def decide_action(self, grid: np.ndarray, available_actions: Optional[List[str]] = None) -> str:
        self.step_count += 1
        
        if available_actions is None:
            available_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'INTERACT']
        
        # Adaptive strategy
        if self.step_count < 5:
            # Initial exploration
            return self._explore(available_actions)
        elif self.consecutive_failures > 3:
            # Stuck - try interaction
            if 'INTERACT' in available_actions:
                return 'INTERACT'
        else:
            # Pattern-based decision
            return self._pattern_based(grid, available_actions)
    
    def _explore(self, actions: List[str]) -> str:
        """Explore actions"""
        # Prefer INTERACT early
        if random.random() < 0.6 and 'INTERACT' in actions:
            return 'INTERACT'
        movements = [a for a in actions if a in ['UP', 'DOWN', 'LEFT', 'RIGHT']]
        if movements:
            return random.choice(movements)
        return random.choice(actions)
    
    def _pattern_based(self, grid: np.ndarray, actions: List[str]) -> str:
        """Make pattern-based decisions"""
        # Count filled cells
        filled = np.sum(grid > 0)
        
        if filled < grid.size * 0.2:  # Grid mostly empty
            if 'INTERACT' in actions and random.random() < 0.5:
                return 'INTERACT'
        
        if 'UP' in actions and random.random() < 0.4:
            return 'UP'
        
        if actions:
            return random.choice(actions)
        return 'WAIT'
    
    def get_stats(self) -> Dict:
        return {'name': self.name, 'steps': self.step_count}

if __name__ == "__main__":
    # Test simplified agent
    agent = SimplifiedARCAgent()
    test_grid = np.random.randint(0, 10, (10, 10))
    
    agent.reset()
    for _ in range(5):
        action = agent.decide_action(test_grid)
        print(f"Action: {action}")
        agent.observe_reward(random.random(), test_grid)
    
    print(f"Stats: {agent.get_stats()}")
