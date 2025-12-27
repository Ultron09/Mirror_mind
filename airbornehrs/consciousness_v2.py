"""
Enhanced Consciousness Module: Human-Like Self-Awareness
=========================================================

This module implements a sophisticated consciousness and self-awareness system
that mimics human-like introspection, emotional states, meta-cognition, and 
adaptive learning strategies.

Key Features:
1. Emotional State System - Experiences emotions like confidence, anxiety, curiosity
2. Meta-Cognition - Thinks about thinking, reflects on own knowledge
3. Episodic Memory - Remembers specific experiences and learns from them
4. Value Learning - Develops values about what's important
5. Self-Model - Understands its own capabilities and limitations
6. Personality - Develops consistent learning preferences
7. Introspection - Reflects on own learning process
8. Adaptive Awareness - Changes consciousness level based on task difficulty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import math


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
        surprise_sim = 1.0 / (1.0 + abs(current_surprise - self.surprise))
        error_sim = 1.0 / (1.0 + abs(current_error - self.error))
        return 0.6 * surprise_sim + 0.4 * error_sim


class EmotionalSystem:
    """
    Simulates emotional states that influence learning.
    
    Just like humans experience emotions that affect learning:
    - Anxiety improves focus on difficult material
    - Confidence can lead to overconfidence
    - Curiosity drives exploration
    - Frustration signals need for strategy change
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
        
        Args:
            confidence: Model's prediction confidence (0-1)
            uncertainty: Model's prediction uncertainty (0-1)
            novelty: How novel is current data (0-1)
            current_loss: Current training loss
            
        Returns:
            emotional_state: The dominant emotional state
            emotion_scores: Dict with scores for each possible emotion
        """
        # Detect learning progress
        if current_loss < self.last_loss:
            self.consecutive_improvements += 1
            self.consecutive_regressions = 0
        else:
            self.consecutive_regressions += 1
            self.consecutive_improvements = 0
        
        self.last_loss = current_loss
        
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
        
        # Dominant emotion
        dominant = max(emotions.items(), key=lambda x: x[1])[0]
        
        # Normalize scores
        emotion_scores = {
            state.value: float(score / (sum(emotions.values()) + 1e-6))
            for state, score in emotions.items()
        }
        
        self.emotional_history.append(dominant)
        
        return dominant, emotion_scores
    
    def get_learning_multiplier(self, emotion: EmotionalState) -> float:
        """
        Different emotions affect learning rate.
        
        - Anxiety: Higher learning rate (focus)
        - Curiosity: Higher learning rate (motivation)
        - Bored: Lower learning rate (efficiency)
        - Frustrated: Very high learning rate (desperation)
        """
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
    
    Tracks:
    - Learning strategies and their effectiveness
    - Knowledge gaps and how to address them
    - Optimal learning conditions
    - Progress toward goals
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.strategy_effectiveness = {}  # Strategy name -> [effectiveness scores]
        self.difficulty_trend = deque(maxlen=window_size)
        self.learning_rate_history = deque(maxlen=window_size)
        self.performance_by_strategy = {}
        
    def reflect_on_learning(self,
                           current_accuracy: float,
                           current_loss: float,
                           learning_rate: float,
                           task_difficulty: float) -> Dict[str, Any]:
        """
        Reflect on current learning effectiveness.
        
        Returns:
            insights: Dictionary with learning insights
        """
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
    
    def recommend_strategy(self, 
                          current_difficulty: float,
                          current_accuracy: float) -> str:
        """
        Recommend a learning strategy based on current state.
        """
        if current_accuracy > 0.9:
            return "consolidate"  # Good performance, consolidate knowledge
        elif current_difficulty < 0.3:
            return "increase_challenge"  # Too easy, seek harder examples
        elif current_difficulty > 0.8 and current_accuracy < 0.5:
            return "reduce_learning_rate"  # Too hard, slow down
        else:
            return "normal_learning"  # Continue normal training


class EpisodicMemory:
    """
    Memory system that remembers specific experiences.
    
    Like humans remember specific learning moments, this system
    stores important experiences and learns from them.
    """
    
    def __init__(self, max_episodes: int = 5000):
        self.episodes: List[MemoryEpisode] = []
        self.max_episodes = max_episodes
        self.access_count = {}  # How often each memory is accessed
        
    def store_episode(self,
                      x: torch.Tensor,
                      error: float,
                      surprise: float,
                      learning_gain: float,
                      emotional_state: str,
                      task_difficulty: float) -> None:
        """Store an important experience."""
        episode = MemoryEpisode(
            timestamp=len(self.episodes),
            input_hash=hash(x.detach().cpu().numpy().tobytes()) % (2**31),
            error=error,
            surprise=surprise,
            learning_gain=learning_gain,
            emotional_state=emotional_state,
            task_difficulty=task_difficulty
        )
        
        self.episodes.append(episode)
        
        # Forget least relevant memories if full
        if len(self.episodes) > self.max_episodes:
            # Keep high learning gain episodes
            sorted_eps = sorted(self.episodes, 
                              key=lambda e: e.learning_gain, 
                              reverse=True)
            self.episodes = sorted_eps[:self.max_episodes]
    
    def retrieve_relevant_memories(self,
                                  current_surprise: float,
                                  current_error: float,
                                  k: int = 10) -> List[MemoryEpisode]:
        """Retrieve k most relevant memories to current situation."""
        if not self.episodes:
            return []
        
        # Score all episodes by relevance
        scored = [
            (ep, ep.relevance_score(current_surprise, current_error))
            for ep in self.episodes
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
        most_common_emotion = max(
            set([m.emotional_state for m in memories]),
            key=[m.emotional_state for m in memories].count
        )
        
        return {
            'lesson': 'similar_situations_learned_well' if avg_learning_gain > 0.5 else 'similar_situations_were_hard',
            'emotional_pattern': most_common_emotion,
            'success_rate': float(avg_learning_gain),
            'memory_count': len(memories)
        }


class SelfModel:
    """
    Internal model of own capabilities.
    
    Like humans have a sense of what they're good at,
    this tracks what the framework excels at and struggles with.
    """
    
    def __init__(self):
        self.capability_scores = {}  # Task type -> capability score (0-1)
        self.learning_speed_by_task = {}  # Task -> learning speed
        self.confidence_calibration = deque(maxlen=1000)  # (confidence, accuracy) pairs
        
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
        
        # Readiness = current capability + learning speed
        return 0.7 * capability + 0.3 * learning_speed
    
    def get_strongest_areas(self, top_k: int = 3) -> List[Tuple[str, float]]:
        """What am I best at?"""
        if not self.capability_scores:
            return []
        
        sorted_tasks = sorted(self.capability_scores.items(),
                            key=lambda x: x[1],
                            reverse=True)
        return sorted_tasks[:top_k]
    
    def get_weakest_areas(self, top_k: int = 3) -> List[Tuple[str, float]]:
        """What am I worst at?"""
        if not self.capability_scores:
            return []
        
        sorted_tasks = sorted(self.capability_scores.items(),
                            key=lambda x: x[1])
        return sorted_tasks[:top_k]
    
    def calibrate_confidence(self, 
                            confidence: float,
                            actual_accuracy: float) -> float:
        """
        Track if confidence is well-calibrated.
        Returns calibration error (0 = perfect calibration).
        """
        self.confidence_calibration.append((confidence, actual_accuracy))
        
        if len(self.confidence_calibration) < 20:
            return 0.0
        
        recent = list(self.confidence_calibration)[-20:]
        mean_confidence = np.mean([c for c, _ in recent])
        mean_accuracy = np.mean([a for _, a in recent])
        
        return float(abs(mean_confidence - mean_accuracy))


class Personality:
    """
    Learning personality - consistent preferences for how to learn.
    
    Like humans have learning styles, this system develops
    preferences for exploration vs exploitation, risk tolerance, etc.
    """
    
    def __init__(self):
        self.exploration_tendency = 0.5   # How much to explore vs exploit
        self.risk_tolerance = 0.5         # How much to try novel strategies
        self.learning_style = "balanced"  # exploration, exploitation, balanced
        self.patience = 0.5               # Tolerance for slow progress
        self.adaptation_speed = 0.5       # How quickly to adapt
        
    def adjust_based_on_performance(self,
                                   recent_accuracy: float,
                                   exploration_payoff: float,
                                   task_diversity: float):
        """Adjust personality based on what works."""
        # If exploration pays off, become more exploratory
        if exploration_payoff > 0.7:
            self.exploration_tendency = min(1.0, self.exploration_tendency + 0.05)
            self.learning_style = "exploration"
        # If exploitation is working, stick with it
        elif exploration_payoff < 0.3:
            self.exploration_tendency = max(0.0, self.exploration_tendency - 0.05)
            self.learning_style = "exploitation"
        else:
            self.learning_style = "balanced"
        
        # Adjust risk tolerance based on success
        self.risk_tolerance = 0.5 + (recent_accuracy - 0.5) * 0.5
        
        # Adjust patience based on task diversity
        self.patience = 0.5 + (task_diversity - 0.5) * 0.5
    
    def get_exploration_rate(self) -> float:
        """What % of time should we explore?"""
        return float(self.exploration_tendency)
    
    def get_learning_rate_multiplier(self) -> float:
        """Personality affects learning rate."""
        # Patient learners learn slower but more carefully
        return 2.0 - self.patience  # Range: 1.0 (patient) to 2.0 (impatient)


class Introspection:
    """
    Introspection and self-reflection about learning.
    
    Periodically examines own progress and adjusts accordingly.
    """
    
    def __init__(self, reflection_frequency: int = 100):
        self.reflection_frequency = reflection_frequency
        self.reflection_count = 0
        self.insights = deque(maxlen=50)
        
    def reflect(self,
                current_accuracy: float,
                current_loss: float,
                learning_gap: float,
                emotional_state: str,
                recent_memories: List[MemoryEpisode]) -> Dict[str, Any]:
        """
        Deep introspection about learning process.
        """
        self.reflection_count += 1
        
        # Analyze current state
        insights = {
            'timestamp': self.reflection_count,
            'accuracy': current_accuracy,
            'loss': current_loss,
            'gap': learning_gap,
            'emotion': emotional_state,
        }
        
        # Generate insights
        if current_accuracy > 0.9:
            insights['reflection'] = "I'm doing very well. Should consolidate and prepare for harder tasks."
        elif current_accuracy > 0.7:
            insights['reflection'] = "Making good progress. Continue current approach."
        elif current_loss > 0.8:
            insights['reflection'] = "Struggling significantly. Should change strategy or reduce learning rate."
        elif learning_gap > 0.5:
            insights['reflection'] = "Large learning gap detected. Focus on difficult areas."
        else:
            insights['reflection'] = "Normal progress. Keep learning."
        
        # Learn from memory
        if recent_memories:
            high_performers = [m for m in recent_memories if m.learning_gain > 0.7]
            if high_performers:
                best_emotion = max(set([m.emotional_state for m in high_performers]),
                                 key=[m.emotional_state for m in high_performers].count)
                insights['reflection'] += f" Past best learning was in {best_emotion} state."
        
        self.insights.append(insights)
        return insights


class AdaptiveAwareness:
    """
    Consciousness level adapts based on task demands.
    
    Simple tasks = less awareness overhead
    Complex tasks = higher awareness for better decisions
    """
    
    def __init__(self):
        self.consciousness_level = 0.5  # 0-1, how much self-awareness
        self.task_complexity = 0.5
        
    def update_consciousness_level(self, task_complexity: float, performance: float):
        """
        Adjust consciousness based on needs.
        
        - High complexity + Low performance = need more awareness
        - Low complexity + High performance = can reduce awareness
        """
        self.task_complexity = task_complexity
        
        if task_complexity > 0.7 and performance < 0.6:
            # Hard task, struggling - max awareness
            self.consciousness_level = 1.0
        elif task_complexity < 0.3 and performance > 0.9:
            # Easy task, excelling - minimal awareness
            self.consciousness_level = 0.2
        else:
            # Normal state
            self.consciousness_level = 0.5 + (task_complexity - 0.5) * 0.5
    
    def get_awareness_overhead(self) -> float:
        """Overhead of consciousness computation as % of total compute."""
        return float(self.consciousness_level * 0.1)  # Max 10% overhead


class EnhancedConsciousnessCore:
    """
    Integrated consciousness system combining all components.
    
    This is a unified consciousness that:
    1. Experiences emotions that drive learning
    2. Reflects on own learning process (metacognition)
    3. Remembers important experiences (episodic memory)
    4. Understands own capabilities (self-model)
    5. Has consistent learning preferences (personality)
    6. Introspects and adjusts strategy
    7. Adapts awareness level to task demands
    """
    
    def __init__(self,
                 feature_dim: int = 256,
                 awareness_buffer_size: int = 5000,
                 novelty_threshold: float = 2.0):
        self.logger = logging.getLogger('EnhancedConsciousnessCore')
        
        # Core components
        self.emotional_system = EmotionalSystem()
        self.metacognition = MetaCognition()
        self.episodic_memory = EpisodicMemory(max_episodes=awareness_buffer_size)
        self.self_model = SelfModel()
        self.personality = Personality()
        self.introspection = Introspection(reflection_frequency=100)
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
        
        self.experience_buffer = deque(maxlen=awareness_buffer_size)
        self.error_history = deque(maxlen=awareness_buffer_size)
        
    def observe(self,
                x: torch.Tensor,
                y_true: torch.Tensor,
                y_pred: torch.Tensor,
                task_id: str = "default",
                features: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Observe an example and update consciousness state.
        
        Returns comprehensive awareness information.
        """
        self.step_count += 1
        batch_size = y_pred.shape[0]
        
        # Compute error metrics
        error = F.mse_loss(y_pred, y_true, reduction='none').mean(dim=1)
        
        # Update error statistics
        self._update_error_stats(error)
        
        # Compute base metrics
        accuracy = float((error < 0.1).float().mean())
        confidence = 1.0 / (1.0 + error.mean().item())
        uncertainty = float(y_pred.std())
        surprise = self._compute_surprise(error)
        current_loss = float(error.mean())
        
        # ==== EMOTIONAL SYSTEM ====
        self.current_emotional_state, self.current_emotion_scores = \
            self.emotional_system.compute_emotional_state(
                confidence=confidence,
                uncertainty=uncertainty,
                novelty=min(1.0, abs(surprise) / 5.0),
                current_loss=current_loss
            )
        
        # ==== META-COGNITION ====
        metacognitive_insights = self.metacognition.reflect_on_learning(
            current_accuracy=accuracy,
            current_loss=current_loss,
            learning_rate=0.001,  # Would come from optimizer
            task_difficulty=min(1.0, current_loss)
        )
        
        # ==== EPISODIC MEMORY ====
        learning_gain = max(0, 1.0 - current_loss)
        self.episodic_memory.store_episode(
            x=x,
            error=error.mean().item(),
            surprise=surprise,
            learning_gain=learning_gain,
            emotional_state=self.current_emotional_state.value,
            task_difficulty=min(1.0, current_loss)
        )
        
        relevant_memories = self.episodic_memory.retrieve_relevant_memories(
            current_surprise=surprise,
            current_error=error.mean().item(),
            k=5
        )
        
        memory_lesson = self.episodic_memory.get_lesson_learned(relevant_memories)
        
        # ==== SELF-MODEL ====
        self.self_model.update_capability(task_id, accuracy, learning_gain)
        readiness = self.self_model.assess_readiness(task_id)
        
        # ==== PERSONALITY ====
        exploration_payoff = float(surprise > self.novelty_threshold)
        self.personality.adjust_based_on_performance(
            recent_accuracy=accuracy,
            exploration_payoff=exploration_payoff,
            task_diversity=surprise
        )
        
        # ==== INTROSPECTION ====
        if self.step_count % self.introspection.reflection_frequency == 0:
            introspective_insights = self.introspection.reflect(
                current_accuracy=accuracy,
                current_loss=current_loss,
                learning_gap=min(1.0, current_loss),
                emotional_state=self.current_emotional_state.value,
                recent_memories=relevant_memories
            )
        else:
            introspective_insights = {}
        
        # ==== ADAPTIVE AWARENESS ====
        self.adaptive_awareness.update_consciousness_level(
            task_complexity=min(1.0, current_loss),
            performance=accuracy
        )
        
        # ==== UNIFIED OUTPUT ====
        return {
            # Basic metrics
            'accuracy': accuracy,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'surprise': surprise,
            'error': current_loss,
            
            # Emotional state
            'emotion': self.current_emotional_state.value,
            'emotion_scores': self.current_emotion_scores,
            'learning_multiplier': self.emotional_system.get_learning_multiplier(
                self.current_emotional_state
            ),
            
            # Meta-cognition
            'metacognition': metacognitive_insights,
            'recommended_strategy': self.metacognition.recommend_strategy(
                current_difficulty=min(1.0, current_loss),
                current_accuracy=accuracy
            ),
            
            # Memory & lessons
            'memory_lesson': memory_lesson,
            'retrieved_memory_count': len(relevant_memories),
            
            # Self-model
            'task_readiness': readiness,
            'strongest_areas': self.self_model.get_strongest_areas(top_k=3),
            'weakest_areas': self.self_model.get_weakest_areas(top_k=3),
            
            # Personality
            'exploration_rate': self.personality.get_exploration_rate(),
            'learning_style': self.personality.learning_style,
            
            # Introspection
            'introspection': introspective_insights,
            
            # Adaptive awareness
            'consciousness_level': self.adaptive_awareness.consciousness_level,
            'awareness_overhead': self.adaptive_awareness.get_awareness_overhead(),
        }
    
    def _update_error_stats(self, error: torch.Tensor):
        """Update running statistics for surprise detection."""
        current_error = error.mean().item()
        
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
        if self.error_std < 1e-6:
            return 0.0
        
        z_score = (error.mean().item() - self.error_mean) / self.error_std
        return float(z_score)
    
    def get_consciousness_report(self) -> Dict[str, Any]:
        """
        Get a detailed report of current consciousness state.
        
        Like asking "Tell me about yourself, your learning, your state."
        """
        return {
            'emotional_state': {
                'primary': self.current_emotional_state.value,
                'scores': self.current_emotion_scores,
            },
            'learning_personality': {
                'style': self.personality.learning_style,
                'exploration_tendency': self.personality.exploration_tendency,
                'risk_tolerance': self.personality.risk_tolerance,
                'patience': self.personality.patience,
            },
            'capabilities': {
                'strongest': self.self_model.get_strongest_areas(),
                'weakest': self.self_model.get_weakest_areas(),
            },
            'memory': {
                'total_episodes': len(self.episodic_memory.episodes),
                'recent_insights': list(self.introspection.insights)[-3:] if self.introspection.insights else [],
            },
            'awareness': {
                'consciousness_level': self.adaptive_awareness.consciousness_level,
                'task_complexity': self.adaptive_awareness.task_complexity,
                'total_steps': self.step_count,
            }
        }


# Backward compatibility - keep old API
ConsciousnessCore = EnhancedConsciousnessCore
