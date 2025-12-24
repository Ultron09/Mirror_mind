"""
Consciousness Layer: Intelligent Attention & Self-Awareness
============================================================
Gives the framework "consciousness" about what to learn and why.

This layer implements:
1. Feature Importance Tracking - learns which features matter most
2. Intrinsic Motivation - identifies learning gaps autonomously  
3. Self-Awareness - tracks knowledge state and confidence
4. Attention Weighting - focuses adaptation on critical dimensions

Philosophy: A conscious system should know:
- What it knows well (confidence high)
- What it doesn't know (uncertainty high)
- What matters most (importance high)
- What changed (surprise high)
And prioritize learning in high-gap areas.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from collections import deque
import math


class ConsciousnessCore:
    """
    Self-aware learning system that tracks knowledge and prioritizes adaptation.
    
    Tracks 4 dimensions of understanding:
    1. Confidence: How certain are predictions? (inverse of error)
    2. Uncertainty: How variable are predictions? (variance across ensemble)
    3. Importance: Which dimensions affect loss most?
    4. Surprise: Is this example out-of-distribution?
    """
    
    def __init__(self, 
                 model: nn.Module,
                 feature_dim: int = 256,
                 awareness_buffer_size: int = 5000,
                 novelty_threshold: float = 2.0):
        """
        Args:
            model: The framework model
            feature_dim: Dimension of representations to analyze
            awareness_buffer_size: How many experiences to track for statistics
            novelty_threshold: Z-score threshold for "new knowledge"
        """
        self.model = model
        self.feature_dim = feature_dim
        self.awareness_buffer_size = awareness_buffer_size
        self.novelty_threshold = novelty_threshold
        self.logger = logging.getLogger('ConsciousnessCore')
        
        # === KNOWLEDGE STATE ===
        # What does the model know about each feature/dimension?
        self.feature_importance = {}  # Per-dimension importance scores
        self.feature_confidence = {}  # Per-dimension prediction confidence
        self.feature_uncertainty = {}  # Per-dimension uncertainty (variance)
        
        # === EXPERIENCE HISTORY ===
        # Rolling buffer of experiences for statistical analysis
        self.experience_buffer = deque(maxlen=awareness_buffer_size)
        self.prediction_history = deque(maxlen=awareness_buffer_size)
        self.error_history = deque(maxlen=awareness_buffer_size)
        
        # === LEARNING GAPS ===
        # Areas where the model struggles or lacks knowledge
        self.learning_gaps = {}  # Dimension -> gap score (0-1)
        self.knowledge_frontiers = []  # Tasks at the edge of knowledge
        
        # === SURPRISE DETECTOR ===
        # Detects out-of-distribution or novel examples
        self.error_mean = 0.0
        self.error_std = 1.0
        self.error_ewma = 0.99  # Exponential moving average weight
    
    def observe(self, 
                x: torch.Tensor, 
                y_true: torch.Tensor, 
                y_pred: torch.Tensor,
                features: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Observe an example and update internal knowledge state.
        
        Args:
            x: Input tensor
            y_true: Ground truth
            y_pred: Model prediction
            features: Optional intermediate representation (e.g., from encoder)
        
        Returns:
            dict with 'confidence', 'uncertainty', 'surprise', 'importance'
        """
        batch_size = y_pred.shape[0]
        
        # Compute error
        error = F.mse_loss(y_pred, y_true, reduction='none').mean(dim=1)
        
        # Update running error statistics for surprise detection
        self._update_error_stats(error)
        
        # Compute metrics
        confidence = 1.0 / (1.0 + error.mean().item())  # Inverse error
        surprise = self._compute_surprise(error)  # Z-score of error
        uncertainty = self._compute_uncertainty(y_pred)  # Variance
        
        # If we have features, compute importance
        importance = 1.0
        if features is not None:
            importance = self._compute_feature_importance(features, error)
        
        # Store in buffers
        self.experience_buffer.append({
            'x': x.detach(),
            'error': error.detach(),
            'surprise': surprise
        })
        self.prediction_history.append(y_pred.detach())
        self.error_history.append(error.detach())
        
        # Update learning gaps (areas needing improvement)
        self._update_learning_gaps(error, surprise)
        
        metrics = {
            'confidence': confidence,
            'uncertainty': uncertainty,
            'surprise': surprise,
            'importance': importance
        }
        
        return metrics
    
    def _update_error_stats(self, error: torch.Tensor):
        """Update running statistics of error for anomaly detection."""
        current_error = error.mean().item()
        
        # Exponential moving average for mean
        self.error_mean = self.error_ewma * self.error_mean + (1 - self.error_ewma) * current_error
        
        # Compute std with running variance - initialized properly
        if not hasattr(self, '_error_variance'):
            self._error_variance = 1.0  # Start with 1.0 instead of 0.0 to avoid freezing
        
        # Variance computed using OLD mean (before update) for Welford's algorithm
        variance_increment = (current_error - self.error_mean) ** 2
        self._error_variance = self.error_ewma * self._error_variance + (1 - self.error_ewma) * variance_increment
        
        # Std dev: must be at least 1e-4 to prevent division by zero in surprise calc
        self.error_std = max(np.sqrt(self._error_variance), 1e-4)
    
    def _compute_surprise(self, error: torch.Tensor) -> float:
        """
        Compute surprise as Z-score of error.
        High surprise = out-of-distribution or novel example.
        """
        if self.error_std < 1e-6:
            return 0.0
        
        z_score = (error.mean().item() - self.error_mean) / self.error_std
        return float(z_score)
    
    def _compute_uncertainty(self, y_pred: torch.Tensor) -> float:
        """
        Compute uncertainty as variance of predictions.
        High uncertainty = model is unsure.
        """
        pred_var = y_pred.var().item()
        pred_std = np.sqrt(pred_var)
        return float(pred_std)
    
    def _compute_feature_importance(self, 
                                   features: torch.Tensor, 
                                   error: torch.Tensor) -> float:
        """
        Compute which features contribute most to error.
        Uses correlation between feature variance and error.
        """
        if features is None or features.shape[0] == 0:
            return 1.0
        
        # Standardize features and errors
        feat_norm = F.normalize(features.view(features.shape[0], -1), p=2, dim=1)
        err_norm = (error - error.mean()) / (error.std() + 1e-6)
        
        # Compute correlation (absolute dot product)
        correlation = torch.abs(feat_norm.mean(dim=0)).mean().item()
        
        # Importance = how much features vary with error
        importance = min(1.0, correlation * 10.0)  # Scale to [0, 1]
        
        return float(importance)
    
    def _update_learning_gaps(self, error: torch.Tensor, surprise: float):
        """
        Identify areas where the model struggles (learning gaps).
        These should be prioritized for learning.
        """
        # High error = clear gap
        gap_from_error = error.mean().item()
        
        # High surprise = novel, unknown territory
        gap_from_surprise = max(0, (surprise - self.novelty_threshold) / self.novelty_threshold)
        
        # Combined gap score
        overall_gap = 0.7 * gap_from_error + 0.3 * gap_from_surprise
        
        # Store as learning gap
        if not hasattr(self, 'current_gap'):
            self.current_gap = overall_gap
        else:
            # Exponential average
            self.current_gap = 0.9 * self.current_gap + 0.1 * overall_gap
    
    def get_learning_priority(self) -> Dict[str, float]:
        """
        Return what the model should prioritize learning.
        
        Returns:
            dict with 'learn_from_gap', 'replay_priority', 'consolidation_urgency'
        """
        gap = getattr(self, 'current_gap', 0.5)
        
        return {
            'learn_from_gap': gap,  # How much to focus on learning gaps
            'replay_priority': min(1.0, gap * 2.0),  # Boost replay of hard examples
            'consolidation_urgency': min(1.0, gap * 1.5)  # Consolidate when gaps are high
        }
    
    def get_knowledge_state(self) -> Dict[str, any]:
        """
        Get a snapshot of what the model knows and doesn't know.
        """
        recent_errors = list(self.error_history)[-100:] if len(self.error_history) > 0 else [0]
        recent_errors = torch.cat(recent_errors, dim=0) if isinstance(recent_errors[0], torch.Tensor) else torch.tensor(recent_errors)
        
        return {
            'confidence': float(1.0 / (1.0 + recent_errors.mean().item())),
            'learning_gap': getattr(self, 'current_gap', 0.5),
            'experience_count': len(self.experience_buffer),
            'error_mean': float(self.error_mean),
            'error_std': float(self.error_std)
        }


class AttentionMechanism(nn.Module):
    """
    Learnable attention over features/dimensions.
    
    The model learns which features matter most for each task.
    This is "consciousness" because it's self-aware about feature importance.
    """
    
    def __init__(self, 
                 feature_dim: int = 256,
                 num_heads: int = 8,
                 learned: bool = True):
        """
        Args:
            feature_dim: Dimension of input features
            num_heads: Number of attention heads
            learned: Whether to learn attention (vs fixed)
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.learned = learned
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"
        
        if learned:
            # Learnable query, key, value projections
            self.q_proj = nn.Linear(feature_dim, feature_dim)
            self.k_proj = nn.Linear(feature_dim, feature_dim)
            self.v_proj = nn.Linear(feature_dim, feature_dim)
            self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention to identify important features.
        
        Args:
            x: Input tensor (batch_size, seq_len, feature_dim) or (batch_size, feature_dim)
            mask: Optional mask (batch_size, seq_len) or None
        
        Returns:
            output: Attended features (same shape as x)
            attention_weights: (batch_size, num_heads, seq_len, seq_len) or (batch_size, num_heads, 1, feature_dim)
        """
        # Handle 2D input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, feature_dim)
        
        batch_size, seq_len, _ = x.shape
        
        if self.learned:
            # Learned attention (self-attention)
            q = self.q_proj(x)  # (batch, seq_len, feature_dim)
            k = self.k_proj(x)
            v = self.v_proj(x)
        else:
            # Fixed feature importance (no learned parameters)
            q = k = v = x
        
        # Split into multiple heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, v)
        
        # Merge heads
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, seq_len, self.feature_dim)
        
        if self.learned:
            output = self.out_proj(attended)
        else:
            output = attended
        
        return output, attention_weights
    
    def get_feature_importance(self) -> torch.Tensor:
        """
        Extract which features are most attended to.
        Returns a (feature_dim,) tensor with importance scores.
        """
        if not self.learned or not hasattr(self, 'out_proj'):
            return torch.ones(self.feature_dim) / self.feature_dim
        
        # Use output projection weights as importance proxy
        importance = torch.abs(self.out_proj.weight).mean(dim=0)
        importance = importance / (importance.sum() + 1e-6)
        
        return importance


class IntrinisicMotivation:
    """
    Drives learning toward areas of uncertainty and novelty.
    
    This is the "curiosity" that makes the system want to learn about things it doesn't know.
    """
    
    def __init__(self, 
                 update_frequency: int = 100,
                 uncertainty_weight: float = 0.5,
                 novelty_weight: float = 0.3,
                 learning_progress_weight: float = 0.2):
        """
        Args:
            update_frequency: How often to update motivation signal
            uncertainty_weight: Weight for "I don't know" signal
            novelty_weight: Weight for "this is new" signal
            learning_progress_weight: Weight for "am I improving?" signal
        """
        self.update_frequency = update_frequency
        self.uncertainty_weight = uncertainty_weight
        self.novelty_weight = novelty_weight
        self.learning_progress_weight = learning_progress_weight
        
        self.step_count = 0
        self.last_loss = float('inf')
        self.loss_history = deque(maxlen=100)
        self.logger = logging.getLogger('IntrinisicMotivation')
    
    def compute_motivation(self, 
                          uncertainty: float, 
                          novelty_zscore: float,
                          current_loss: float) -> float:
        """
        Compute intrinsic motivation signal.
        
        Args:
            uncertainty: Model's prediction variance (0-1)
            novelty_zscore: How novel is this example? (0-inf)
            current_loss: Current training loss
        
        Returns:
            motivation: 0-1 score, higher = model should learn more
        """
        # Signal 1: Uncertainty (I don't know)
        uncertainty_signal = min(1.0, uncertainty * 2.0)
        
        # Signal 2: Novelty (this is new to me)
        novelty_signal = min(1.0, novelty_zscore / 4.0)
        
        # Signal 3: Learning progress (am I getting better?)
        self.loss_history.append(current_loss)
        if len(self.loss_history) > 1:
            loss_improvement = (self.last_loss - current_loss) / (self.last_loss + 1e-6)
            learning_progress_signal = max(0, -loss_improvement)  # Negative progress = motivation to learn more
        else:
            learning_progress_signal = 0.5
        
        # Combine signals
        motivation = (
            self.uncertainty_weight * uncertainty_signal +
            self.novelty_weight * novelty_signal +
            self.learning_progress_weight * learning_progress_signal
        )
        
        self.last_loss = current_loss
        self.step_count += 1
        
        return float(min(1.0, motivation))


class SelfAwarenessMonitor:
    """
    Tracks what the framework knows about itself.
    
    Monitors:
    - Overall learning progress
    - Confidence in different areas
    - Knowledge distribution (what's well-learned vs novel)
    """
    
    def __init__(self, moving_average_window: int = 100):
        self.moving_average_window = moving_average_window
        self.logger = logging.getLogger('SelfAwarenessMonitor')
        
        # Historical tracking
        self.confidence_history = deque(maxlen=moving_average_window)
        self.accuracy_history = deque(maxlen=moving_average_window)
        self.learning_gap_history = deque(maxlen=moving_average_window)
        self.surprise_history = deque(maxlen=moving_average_window)
    
    def update(self,
               confidence: float,
               accuracy: float,
               learning_gap: float,
               surprise: float):
        """Update self-awareness with new observations."""
        self.confidence_history.append(confidence)
        self.accuracy_history.append(accuracy)
        self.learning_gap_history.append(learning_gap)
        self.surprise_history.append(surprise)
    
    def get_status(self) -> Dict[str, float]:
        """Get current self-awareness status."""
        if not self.accuracy_history:
            return {
                'avg_confidence': 0.5,
                'avg_accuracy': 0.0,
                'avg_learning_gap': 0.5,
                'avg_surprise': 0.0,
                'overall_competence': 0.0
            }
        
        avg_confidence = np.mean(list(self.confidence_history))
        avg_accuracy = np.mean(list(self.accuracy_history))
        avg_gap = np.mean(list(self.learning_gap_history))
        avg_surprise = np.mean(list(self.surprise_history))
        
        # Overall competence = confidence + accuracy - gap
        competence = (avg_confidence + avg_accuracy - avg_gap) / 2.0
        competence = max(0, min(1.0, competence))
        
        return {
            'avg_confidence': float(avg_confidence),
            'avg_accuracy': float(avg_accuracy),
            'avg_learning_gap': float(avg_gap),
            'avg_surprise': float(avg_surprise),
            'overall_competence': float(competence)
        }
    
    def should_consolidate(self) -> bool:
        """Decides if framework should consolidate knowledge."""
        if len(self.learning_gap_history) < 10:
            return False
        
        recent_gap = np.mean(list(self.learning_gap_history)[-10:])
        recent_confidence = np.mean(list(self.confidence_history)[-10:])
        
        # Consolidate if confidence is high but gaps are emerging
        return recent_confidence > 0.7 and recent_gap > 0.3
    
    def should_explore(self) -> bool:
        """Decides if framework should explore new areas."""
        if len(self.surprise_history) < 10:
            return True
        
        recent_surprise = np.mean(list(self.surprise_history)[-10:])
        
        # Explore if surprises are high (knowledge frontiers)
        return recent_surprise > 1.5
