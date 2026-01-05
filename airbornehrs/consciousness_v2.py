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
4. OPTIMIZATION: Added Content-Aware Retrieval using feature similarity.
5. SELF-MODEL: Added Fingerprint-Based Task Prediction (Exp 3).
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
    x: Optional[torch.Tensor] = None
    y: Optional[torch.Tensor] = None
    features: Optional[torch.Tensor] = None
    
    def relevance_score(self, current_surprise: float, current_error: float, current_features: Optional[torch.Tensor] = None) -> float:
        """How relevant is this past experience to the current situation?"""
        # Similar situations are more relevant
        # Added epsilon to prevent division by zero
        surprise_sim = 1.0 / (1.0 + abs(current_surprise - self.surprise) + 1e-6)
        error_sim = 1.0 / (1.0 + abs(current_error - self.error) + 1e-6)
        
        content_sim = 0.0
        if current_features is not None and self.features is not None:
            # Cosine similarity for content (Task Context)
            # Use MEAN feature vector to be robust to batch order/size
            v1 = current_features.float().mean(dim=0).view(-1)
            v2 = self.features.float().mean(dim=0).view(-1)
            
            if v1.shape == v2.shape:
                content_sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
                # Normalize to 0-1 range
                content_sim = (content_sim + 1) / 2
        
        # Weighted sum: Content is KING for few-shot learning
        return 0.7 * content_sim + 0.2 * surprise_sim + 0.1 * error_sim


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
            EmotionalState.ANXIOUS: 1.2,
            EmotionalState.CURIOUS: 1.1,
            EmotionalState.BORED: 0.8,
            EmotionalState.FRUSTRATED: 1.5,
            EmotionalState.SATISFIED: 1.0,
            EmotionalState.OVERWHELMED: 0.6,
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
            'learning_rate_appropriate': 0.0001 <= learning_rate_trend <= 0.01,
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
                      task_difficulty: float,
                      y: Optional[torch.Tensor] = None,
                      features: Optional[torch.Tensor] = None) -> None:
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
            task_difficulty=task_difficulty,
            x=x.detach().cpu() if isinstance(x, torch.Tensor) else None,
            y=y.detach().cpu() if y is not None else None,
            features=features.detach().cpu() if features is not None else None
        )
        
        self.episodes.append(episode)
        
        # Forget least relevant memories if full
        if len(self.episodes) > self.max_episodes:
            # Drop the oldest, least effective memory (Simple heuristic)
            self.episodes.pop(0)
    
    def retrieve_relevant_memories(self,
                                  current_surprise: float,
                                  current_error: float,
                                  current_features: Optional[torch.Tensor] = None,
                                  k: int = 10) -> List[MemoryEpisode]:
        """Retrieve k most relevant memories to current situation."""
        if not self.episodes:
            return []
        
        # FULL SCAN: 5000 items is fast enough in Python (<1ms)
        # We do not sample anymore to ensure we find the *best* match.
        candidates = self.episodes

        # Score episodes by relevance
        scored = [
            (ep, ep.relevance_score(current_surprise, current_error, current_features))
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
        self.task_fingerprints = {} # Task -> Tensor fingerprint
        
    def update_capability(self, task_id: str, accuracy: float, learning_speed: float, fingerprint: Optional[torch.Tensor] = None):
        """Update understanding of capability in a task."""
        self.capability_scores[task_id] = accuracy
        self.learning_speed_by_task[task_id] = learning_speed
        if fingerprint is not None:
            self.task_fingerprints[task_id] = fingerprint.detach().cpu()
    
    def assess_readiness(self, task_id: str, fingerprint: Optional[torch.Tensor] = None) -> float:
        """How ready is the model for a new task?"""
        # 1. Exact Match
        if task_id in self.capability_scores:
            capability = self.capability_scores[task_id]
            learning_speed = self.learning_speed_by_task.get(task_id, 0.5)
            return 0.7 * capability + 0.3 * learning_speed
            
        # 2. Fingerprint Similarity (Generalization)
        if fingerprint is not None and self.task_fingerprints:
            try:
                # Calculate similarity with all known tasks
                similarities = []
                target_fp = fingerprint.view(-1).float()
                
                for known_id, known_fp in self.task_fingerprints.items():
                    known_fp_flat = known_fp.view(-1).float().to(target_fp.device)
                    sim = F.cosine_similarity(target_fp.unsqueeze(0), known_fp_flat.unsqueeze(0)).item()
                    similarities.append((known_id, sim))
                
                # Sort by similarity
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                # Weighted average of top-3 similar tasks
                top_k = similarities[:3]
                total_weight = sum(max(0, s[1]) for s in top_k) + 1e-6
                
                weighted_score = 0.0
                for known_id, sim in top_k:
                    weight = max(0, sim) / total_weight
                    known_score = self.capability_scores.get(known_id, 0.5)
                    weighted_score += weight * known_score
                    
                return weighted_score
                
            except Exception as e:
                # Fallback on error
                pass

        # 3. Fallback (Average of known tasks)
        if self.capability_scores:
            return sum(self.capability_scores.values()) / len(self.capability_scores)
            
        return 0.5  # Unknown task

    def propose_curriculum(self, available_tasks: Dict[str, torch.Tensor], curiosity_weight: float = 0.5) -> str:
        """
        Select the best task from a menu based on Readiness and Curiosity.
        Policy: Maximize (Readiness * (1 + Curiosity_Bonus))
        - We want tasks we are 'Ready' for (High Readiness).
        - But we also want tasks that are 'Novel' (High Curiosity).
        """
        best_task = None
        best_score = -1.0
        
        for task_id, fingerprint in available_tasks.items():
            # 1. Readiness (0.0 to 1.0)
            readiness = self.assess_readiness(task_id, fingerprint)
            
            # 2. Curiosity (Novelty)
            # Calculate distance to nearest known task
            min_dist = 1.0
            if self.task_fingerprints:
                target_fp = fingerprint.view(-1).float()
                dists = []
                for known_fp in self.task_fingerprints.values():
                    known_fp_flat = known_fp.view(-1).float().to(target_fp.device)
                    # Cosine distance = 1 - similarity
                    dist = 1.0 - F.cosine_similarity(target_fp.unsqueeze(0), known_fp_flat.unsqueeze(0)).item()
                    dists.append(max(0, dist))
                if dists:
                    min_dist = min(dists)
            
            # Curiosity is high if the task is different from what we know
            curiosity = min_dist 
            
            # 3. Combined Score
            # We target a "Zone of Proximal Development":
            # - Readiness should be high enough (>0.3) to learn.
            # - Curiosity adds a bonus.
            
            # If readiness is too low, we might fail completely, so penalize hard.
            penalty = 1.0 if readiness > 0.2 else 0.1
            
            score = (readiness * (1 + curiosity * curiosity_weight)) * penalty
            
            if score > best_score:
                best_score = score
                best_task = task_id
                
        return best_task if best_task else random.choice(list(available_tasks.keys()))


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


class GlobalWorkspace(nn.Module):
    """
    The 'Theater of Consciousness'.
    A shared working memory that integrates and broadcasts information.
    """
    def __init__(self, dim=256, num_slots=8, num_heads=4):
        super().__init__()
        self.dim = dim
        self.slots = nn.Parameter(torch.randn(1, num_slots, dim))
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.memory_gate = nn.GRUCell(dim, dim) # Recurrence
        
    def forward(self, inputs: torch.Tensor, prev_slots: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process inputs through the workspace.
        inputs: [B, N, D] or [B, D]
        """
        if inputs.dim() == 2: inputs = inputs.unsqueeze(1)
        B = inputs.size(0)
        
        # Initialize or use previous state
        if prev_slots is None:
            slots = self.slots.expand(B, -1, -1) # [B, S, D]
        else:
            slots = prev_slots
            
        # 1. Competition / Read (Slots attend to Inputs)
        # Query: Slots, Key/Value: Inputs
        # "What is important right now?"
        attn_out, _ = self.attention(slots, inputs, inputs)
        slots = self.norm(slots + attn_out)
        
        # 2. Recurrence (Memory Update)
        # Update each slot independently using GRU
        # Reshape to [B*S, D] for GRUCell
        slots_flat = slots.view(-1, self.dim)
        slots_new = self.memory_gate(slots_flat, slots_flat) # Simple self-recurrence
        slots = slots_new.view(B, -1, self.dim)
        
        return slots


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
        
        # Global Workspace (The "Mind's Eye")
        self.global_workspace = GlobalWorkspace(dim=feature_dim)
        self.current_thought = None # Current state of workspace
        
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

    def observe(self, *input_args, y_true, y_pred, task_id: str = "default", features: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Observe an example and update consciousness state.
        Accepts multiple input tensors via *input_args.
        """
        self.step_count += 1
        x = input_args[0] if input_args else torch.tensor([])

        # 1. Error and Surprise
        y_pred_flat = y_pred.view(y_pred.size(0), -1)
        if y_true.dim() == 1 and y_pred_flat.size(1) > 1:
             # Classification
             error = F.cross_entropy(y_pred_flat, y_true, reduction='none')
             probs = F.softmax(y_pred_flat, dim=1)
             confidence, _ = probs.max(dim=1)
             confidence = confidence.mean().item()
             
             # Uncertainty = Entropy
             uncertainty = -(probs * torch.log(probs + 1e-6)).sum(dim=1).mean().item()
             
        else:
             # Regression
             error = F.mse_loss(y_pred_flat, y_true.view_as(y_pred_flat), reduction='none').mean(dim=1)
             confidence = 1.0 / (1.0 + error.mean().item())
             uncertainty = features.var(dim=0).mean().item() if features is not None else 0.5

        surprise = self._compute_surprise(error)
        self._update_error_stats(error)
        
        # 2. Uncertainty and Novelty (from features)
        novelty = 0.0
        if features is not None:
            # Simplified novelty: distance from mean feature vector
            if not hasattr(self, 'mean_feature_vector'):
                self.mean_feature_vector = torch.zeros_like(features.mean(dim=0))
            
            novelty = torch.norm(features.mean(dim=0) - self.mean_feature_vector).item()
            # Update mean feature vector
            self.mean_feature_vector = 0.99 * self.mean_feature_vector + 0.01 * features.mean(dim=0)

            # [V7.3] Process Thought (Global Workspace)
            # We treat the features as inputs to the workspace
            try:
                # Ensure features match workspace dim. If not, project?
                # For now assuming feature_dim matches.
                if features.size(-1) == self.feature_dim:
                    self.current_thought = self.global_workspace(features, self.current_thought)
            except Exception:
                pass # Dimension mismatch or other error, skip thought


        # 3. Meta-Cognition Reflection
        meta_stats = self.metacognition.reflect_on_learning(
            current_accuracy=confidence, # Proxy
            current_loss=error.mean().item(),
            learning_rate=0.001, # Placeholder
            task_difficulty=uncertainty
        )
        
        # 4. Emotional State
        current_loss = error.mean().item()
        self.current_emotional_state, self.current_emotion_scores = self.emotional_system.compute_emotional_state(
            confidence=confidence,
            uncertainty=uncertainty,
            novelty=novelty,
            current_loss=current_loss
        )
        
        # 5. Store Episode
        self.episodic_memory.store_episode(
            x=x,
            error=current_loss,
            surprise=surprise,
            learning_gain=self.emotional_system.consecutive_improvements,
            emotional_state=self.current_emotional_state.value,
            task_difficulty=uncertainty, # Simple heuristic
            y=y_true,
            features=features
        )

        # 6. Return Metrics
        return {
            'loss': current_loss,
            'surprise': surprise,
            'uncertainty': uncertainty,
            'confidence': confidence,
            'novelty': novelty,
            'emotion': self.current_emotional_state.value,
            'emotion_scores': self.current_emotion_scores,
            'importance': 1.0 + surprise + (1-confidence),
            'learning_rate_multiplier': self.emotional_system.get_learning_multiplier(self.current_emotional_state)
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
