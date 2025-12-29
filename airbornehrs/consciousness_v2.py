"""
Enhanced Consciousness Module: Human-Like Self-Awareness (Universal V7.2)
=========================================================================

This module implements a sophisticated consciousness and self-awareness system
that mimics human-like introspection, emotional states, meta-cognition, and 
adaptive learning strategies.

PATCH NOTES (V7.2):
1. ACCURACY: Removed random sampling in memory retrieval (Scan all 5k items).
2. STABILITY: Added robust NaN guards in EmotionalSystem.
3. TYPE SAFETY: Fixed Enum/String serialization in JSON outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
import random


class EmotionalState(Enum):
    """Emotional states that drive learning behavior."""
    CONFIDENT = "confident"      # High competence, low uncertainty
    ANXIOUS = "anxious"          # High uncertainty, low competence
    CURIOUS = "curious"          # High novelty, high uncertainty
    BORED = "bored"              # Low novelty, high competence
    FRUSTRATED = "frustrated"    # High effort, low progress
    SATISFIED = "satisfied"      # Making progress, low error
    OVERWHELMED = "overwhelmed"  # High uncertainty, high task complexity


@dataclass
class MemoryEpisode:
    """An episodic memory entry - a specific experience the model learned from."""
    timestamp: int
    input_hash: int
    error: float
    surprise: float
    learning_gain: float
    emotional_state: str
    task_difficulty: float
    features: Optional[torch.Tensor] = None
    
    def relevance_score(self, current_surprise: float, current_error: float) -> float:
        """How relevant is this past experience to the current situation?"""
        # Similar situations are more relevant
        # Added epsilon to prevent division by zero
        surprise_sim = 1.0 / (1.0 + abs(current_surprise - self.surprise) + 1e-6)
        error_sim = 1.0 / (1.0 + abs(current_error - self.error) + 1e-6)
        return 0.6 * surprise_sim + 0.4 * error_sim


class EmotionalSystem:
    """
    Simulates emotional states that influence learning.
    """
    
    def __init__(self,
                 confidence_weight: float = 0.4,
                 uncertainty_weight: float = 0.3,
                 novelty_weight: float = 0.2,
                 progress_weight: float = 0.1):
        self.confidence_weight = confidence_weight
        self.uncertainty_weight = uncertainty_weight
        self.novelty_weight = novelty_weight
        self.progress_weight = progress_weight
        
        self.emotional_history = deque(maxlen=100)
        self.last_loss = float('inf')
        self.consecutive_improvements = 0
        self.consecutive_regressions = 0
        
    def compute_emotional_state(self,
                                confidence: float,
                                uncertainty: float,
                                novelty: float,
                                current_loss: float) -> Tuple[EmotionalState, Dict[str, float]]:
        """
        Compute emotional state based on current metrics.
        """
        # Detect learning progress
        if current_loss < self.last_loss:
            self.consecutive_improvements += 1
            self.consecutive_regressions = 0
        else:
            self.consecutive_regressions += 1
            self.consecutive_improvements = 0
        
        self.last_loss = current_loss
        
        # Guard against NaNs in inputs (CRITICAL FIX)
        confidence = 0.0 if math.isnan(confidence) else confidence
        uncertainty = 1.0 if math.isnan(uncertainty) else uncertainty
        novelty = 0.0 if math.isnan(novelty) else novelty
        
        # Compute emotion scores
        emotions = {
            EmotionalState.CONFIDENT: confidence * (1 - uncertainty) * (1 - novelty),
            EmotionalState.ANXIOUS: uncertainty * (1 - confidence),
            EmotionalState.CURIOUS: novelty * uncertainty,
            EmotionalState.BORED: (1 - novelty) * confidence,
            EmotionalState.FRUSTRATED: float(self.consecutive_regressions > 5) * (1 - confidence),
            EmotionalState.SATISFIED: float(self.consecutive_improvements > 3) * (1 - uncertainty),
            EmotionalState.OVERWHELMED: uncertainty * novelty * (1 - confidence),
        }
        
        # Safe dominant determination
        try:
            dominant = max(emotions.items(), key=lambda x: x[1])[0]
        except Exception:
            dominant = EmotionalState.CONFIDENT

        # Normalize scores safely
        total_score = sum(emotions.values()) + 1e-6
        emotion_scores = {
            state.value: float(score / total_score)
            for state, score in emotions.items()
        }
        
        self.emotional_history.append(dominant)
        
        return dominant, emotion_scores
    
    def get_learning_multiplier(self, emotion: EmotionalState) -> float:
        """Different emotions affect learning rate."""
        multipliers = {
            EmotionalState.CONFIDENT: 1.0,
            EmotionalState.ANXIOUS: 1.4,         # Focus boost
            EmotionalState.CURIOUS: 1.3,         # Motivation boost
            EmotionalState.BORED: 0.7,           # Don't waste effort
            EmotionalState.FRUSTRATED: 1.8,      # Desperate learning
            EmotionalState.SATISFIED: 1.0,       # Normal pace
            EmotionalState.OVERWHELMED: 0.5,     # Reduce to avoid divergence
        }
        return multipliers.get(emotion, 1.0)


class MetaCognition:
    """
    Thinking about thinking - understanding one's own learning process.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.strategy_effectiveness = {}
        self.difficulty_trend = deque(maxlen=window_size)
        self.learning_rate_history = deque(maxlen=window_size)
        
    def reflect_on_learning(self,
                           current_accuracy: float,
                           current_loss: float,
                           learning_rate: float,
                           task_difficulty: float) -> Dict[str, Any]:
        """Reflect on current learning effectiveness."""
        self.difficulty_trend.append(task_difficulty)
        self.learning_rate_history.append(learning_rate)
        
        # Analyze trends
        if len(self.difficulty_trend) > 10:
            difficulty_trend = np.mean(list(self.difficulty_trend)[-10:])
            learning_rate_trend = np.mean(list(self.learning_rate_history)[-10:])
        else:
            difficulty_trend = task_difficulty
            learning_rate_trend = learning_rate
        
        # Determine learning strategy effectiveness
        is_learning = current_loss < 0.5  # Simple heuristic
        
        return {
            'is_learning_effectively': is_learning,
            'difficulty_increasing': difficulty_trend > 0.5,
            'learning_rate_appropriate': 0.001 <= learning_rate_trend <= 0.1,
            'should_adjust_strategy': not is_learning and task_difficulty > 0.7,
            'current_accuracy': float(current_accuracy),
            'training_efficiency': float(current_accuracy / (current_loss + 1e-6))
        }


class EpisodicMemory:
    """
    Memory system that remembers specific experiences.
    """
    
    def __init__(self, max_episodes: int = 5000):
        self.episodes: List[MemoryEpisode] = []
        self.max_episodes = max_episodes
        
    def store_episode(self,
                      x: torch.Tensor,
                      error: float,
                      surprise: float,
                      learning_gain: float,
                      emotional_state: str,
                      task_difficulty: float) -> None:
        """
        Store an important experience.
        """
        # OPTIMIZATION: Use random ID instead of expensive content hashing
        episode_id = random.getrandbits(31)
        
        episode = MemoryEpisode(
            timestamp=len(self.episodes),
            input_hash=episode_id,
            error=error,
            surprise=surprise,
            learning_gain=learning_gain,
            emotional_state=emotional_state,
            task_difficulty=task_difficulty
        )
        
        self.episodes.append(episode)
        
        # Forget least relevant memories if full
        if len(self.episodes) > self.max_episodes:
            # Drop the oldest, least effective memory (Simple heuristic)
            self.episodes.pop(0)
    
    def retrieve_relevant_memories(self,
                                  current_surprise: float,
                                  current_error: float,
                                  k: int = 10) -> List[MemoryEpisode]:
        """Retrieve k most relevant memories to current situation."""
        if not self.episodes:
            return []
        
        # FULL SCAN: 5000 items is fast enough in Python (<1ms)
        # We do not sample anymore to ensure we find the *best* match.
        candidates = self.episodes

        # Score episodes by relevance
        scored = [
            (ep, ep.relevance_score(current_surprise, current_error))
            for ep in candidates
        ]
        
        # Return top k
        top_k = sorted(scored, key=lambda x: x[1], reverse=True)[:k]
        return [ep for ep, _ in top_k]
    
    def get_lesson_learned(self, 
                          memories: List[MemoryEpisode]) -> Dict[str, Any]:
        """Extract lessons from retrieved memories."""
        if not memories:
            return {'lesson': 'no_previous_experience'}
        
        avg_learning_gain = np.mean([m.learning_gain for m in memories])
        states = [m.emotional_state for m in memories]
        if states:
            most_common_emotion = max(set(states), key=states.count)
        else:
            most_common_emotion = "neutral"
        
        return {
            'lesson': 'similar_situations_learned_well' if avg_learning_gain > 0.5 else 'similar_situations_were_hard',
            'emotional_pattern': most_common_emotion,
            'success_rate': float(avg_learning_gain),
            'memory_count': len(memories)
        }


class SelfModel:
    """Internal model of own capabilities."""
    
    def __init__(self):
        self.capability_scores = {}  # Task type -> capability score (0-1)
        self.learning_speed_by_task = {}  # Task -> learning speed
        
    def update_capability(self, task_id: str, accuracy: float, learning_speed: float):
        """Update understanding of capability in a task."""
        self.capability_scores[task_id] = accuracy
        self.learning_speed_by_task[task_id] = learning_speed
    
    def assess_readiness(self, task_id: str) -> float:
        """How ready is the model for a new task?"""
        if task_id not in self.capability_scores:
            return 0.5  # Unknown task
        
        capability = self.capability_scores[task_id]
        learning_speed = self.learning_speed_by_task.get(task_id, 0.5)
        
        return 0.7 * capability + 0.3 * learning_speed


class Personality:
    """Learning personality - consistent preferences for how to learn."""
    
    def __init__(self):
        self.exploration_tendency = 0.5   
        self.risk_tolerance = 0.5         
        self.learning_style = "balanced"  
        self.patience = 0.5               
        
    def adjust_based_on_performance(self,
                                   recent_accuracy: float,
                                   exploration_payoff: float,
                                   task_diversity: float):
        # If exploration pays off, become more exploratory
        if exploration_payoff > 0.7:
            self.exploration_tendency = min(1.0, self.exploration_tendency + 0.05)
            self.learning_style = "exploration"
        elif exploration_payoff < 0.3:
            self.exploration_tendency = max(0.0, self.exploration_tendency - 0.05)
            self.learning_style = "exploitation"
        else:
            self.learning_style = "balanced"
        
        self.risk_tolerance = 0.5 + (recent_accuracy - 0.5) * 0.5
        self.patience = 0.5 + (task_diversity - 0.5) * 0.5


class AdaptiveAwareness:
    """Consciousness level adapts based on task demands."""
    
    def __init__(self):
        self.consciousness_level = 0.5 
        self.task_complexity = 0.5
        
    def update_consciousness_level(self, task_complexity: float, performance: float):
        self.task_complexity = task_complexity
        
        if task_complexity > 0.7 and performance < 0.6:
            self.consciousness_level = 1.0 # Maximum Alertness
        elif task_complexity < 0.3 and performance > 0.9:
            self.consciousness_level = 0.2 # Flow State / Autopilot
        else:
            self.consciousness_level = 0.5 + (task_complexity - 0.5) * 0.5


class EnhancedConsciousnessCore:
    """
    Integrated consciousness system combining all components.
    Optimized for low-latency observation (V7.2).
    """
    
    def __init__(self,
                 feature_dim: int = 256,
                 awareness_buffer_size: int = 5000,
                 novelty_threshold: float = 2.0,
                 model: Optional[nn.Module] = None):
        self.logger = logging.getLogger('EnhancedConsciousnessCore')
        
        # Core components
        self.emotional_system = EmotionalSystem()
        self.metacognition = MetaCognition()
        self.episodic_memory = EpisodicMemory(max_episodes=awareness_buffer_size)
        self.self_model = SelfModel()
        self.personality = Personality()
        self.adaptive_awareness = AdaptiveAwareness()
        
        # Basic tracking
        self.feature_dim = feature_dim
        self.novelty_threshold = novelty_threshold
        self.error_mean = 0.0
        self.error_std = 1.0
        self.error_ewma = 0.99
        
        # State tracking
        self.step_count = 0
        self.current_emotional_state = EmotionalState.CONFIDENT
        self.current_emotion_scores = {}
        
        self.learning_priority = {'consolidation_urgency': 0.0, 'replay_priority': 0.5}

    def observe(self,
                x: torch.Tensor,
                y_true: torch.Tensor,
                y_pred: torch.Tensor,
                task_id: str = "default",
                features: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Observe an example and update consciousness state.
        """
        self.step_count += 1
        
        # No-grad for all consciousness math to avoid graph bloating
        with torch.no_grad():
            # Compute error metrics
            if y_pred.shape != y_true.shape:
                # Fallback for mismatched outputs
                try:
                    y_pred_flat = y_pred.view(y_pred.size(0), -1)
                    y_true_flat = y_true.view(y_true.size(0), -1)
                    error = F.mse_loss(y_pred_flat, y_true_flat, reduction='none').mean(dim=1)
                except Exception:
                    error = torch.tensor([1.0], device=y_pred.device)
            else:
                error = F.mse_loss(y_pred, y_true, reduction='none')
                if error.ndim > 1: error = error.mean(dim=tuple(range(1, error.ndim)))
            
            # Update error statistics
            self._update_error_stats(error)
            
            # Compute base metrics
            current_loss = float(error.mean().item())
            # Simple accuracy heuristic for regression/classification general case
            accuracy = float((error < 0.1).float().mean().item()) 
            
            confidence = 1.0 / (1.0 + current_loss)
            uncertainty = float(y_pred.std().item()) if y_pred.numel() > 1 else 0.0
            surprise = self._compute_surprise(error)
            
            # ==== EMOTIONAL SYSTEM ====
            self.current_emotional_state, self.current_emotion_scores = \
                self.emotional_system.compute_emotional_state(
                    confidence=confidence,
                    uncertainty=uncertainty,
                    novelty=min(1.0, abs(surprise) / 5.0),
                    current_loss=current_loss
                )
            
            # ==== EPISODIC MEMORY ====
            learning_gain = max(0, 1.0 - current_loss)
            
            self.episodic_memory.store_episode(
                x=x,
                error=current_loss,
                surprise=surprise,
                learning_gain=learning_gain,
                emotional_state=self.current_emotional_state.value,
                task_difficulty=min(1.0, current_loss)
            )
            
            relevant_memories = self.episodic_memory.retrieve_relevant_memories(
                current_surprise=surprise,
                current_error=current_loss,
                k=5
            )
            
            memory_lesson = self.episodic_memory.get_lesson_learned(relevant_memories)
            
            # ==== SELF-MODEL ====
            self.self_model.update_capability(task_id, accuracy, learning_gain)
            
            # ==== PERSONALITY ====
            exploration_payoff = float(surprise > self.novelty_threshold)
            self.personality.adjust_based_on_performance(
                recent_accuracy=accuracy,
                exploration_payoff=exploration_payoff,
                task_diversity=surprise
            )
            
            # ==== ADAPTIVE AWARENESS ====
            self.adaptive_awareness.update_consciousness_level(
                task_complexity=min(1.0, current_loss),
                performance=accuracy
            )

            # Update Priority
            self.learning_priority['consolidation_urgency'] = 1.0 if surprise > 3.0 else 0.0
            self.learning_priority['replay_priority'] = 0.8 if current_loss > 0.5 else 0.2
            
            # ==== UNIFIED OUTPUT ====
            return {
                'accuracy': accuracy,
                'confidence': confidence,
                'uncertainty': uncertainty,
                'surprise': surprise,
                'error': current_loss,
                'importance': (abs(surprise) + learning_gain) / 2.0,
                'emotion': self.current_emotional_state.value,
                'memory_lesson': memory_lesson,
                'learning_multiplier': self.emotional_system.get_learning_multiplier(self.current_emotional_state)
            }
    
    def _update_error_stats(self, error: torch.Tensor):
        """Update running statistics for surprise detection."""
        current_error = error.mean().item()
        if math.isnan(current_error): return

        self.error_mean = self.error_ewma * self.error_mean + \
                         (1 - self.error_ewma) * current_error
        
        if not hasattr(self, '_error_variance'):
            self._error_variance = 1.0
        
        variance_increment = (current_error - self.error_mean) ** 2
        self._error_variance = self.error_ewma * self._error_variance + \
                              (1 - self.error_ewma) * variance_increment
        
        self.error_std = max(np.sqrt(self._error_variance), 1e-4)
    
    def _compute_surprise(self, error: torch.Tensor) -> float:
        """Compute how surprised the model is by this example."""
        if self.error_std < 1e-6: return 0.0
        val = error.mean().item()
        if math.isnan(val): return 0.0
        
        z_score = (val - self.error_mean) / self.error_std
        return float(z_score)

# Backward compatibility for V7.0 imports
ConsciousnessCore = EnhancedConsciousnessCore