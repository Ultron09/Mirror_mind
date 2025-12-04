"""
airbornehrs: Production-ready adaptive meta-learning framework
==============================================================

A lightweight Python package enabling continuous model learning and improvement
in production systems through adaptive optimization cycles and online meta-learning.

Key Components:
    - AdaptiveFramework: Base learner with introspection hooks
    - MetaController: Adaptation layer for online learning
    - ProductionAdapter: Simplified API for inference with online learning

Quick start:
    >>> from airbornehrs import AdaptiveFramework, MetaController
    >>> framework = AdaptiveFramework(model=your_model)
    >>> controller = MetaController(framework)
    >>> # In your training/inference loop:
    >>> controller.adapt(loss=loss, gradients=grads, metrics=perf_metrics)

Terminology reference (replace marketing terms with research terms):
    AGI                           → Adaptive Meta-Learning System
    Consciousness                 → Recursive State Monitoring / Introspection
    Self-Awareness                → Performance Calibration / Uncertainty Estimation
    Thinking / Reasoning          → Inference / Chain-of-Thought Processing
    Memories / Episodic Memory    → Experience Replay Buffer
    Dreaming                      → Generative Replay / Latent Sampling
    Revolutionary / Magic         → Novel / Proposed / Heuristic
    Stabilizer / Suppressor       → Meta-Controller / Regularizer
    The Loop                      → The Optimization Cycle
    Confidence                    → Logit Probability / Softmax Entropy
    Intuition                     → Learned Heuristic / Implicit Bias
"""

__version__ = "0.1.0"
__author__ = "AirborneHRS Contributors"

# Lazy imports to handle circular dependencies
def __getattr__(name):
    if name == 'AdaptiveFramework':
        from .core import AdaptiveFramework
        return AdaptiveFramework
    elif name == 'AdaptiveFrameworkConfig':
        from .core import AdaptiveFrameworkConfig
        return AdaptiveFrameworkConfig
    elif name == 'IntrospectionModule':
        from .core import IntrospectionModule
        return IntrospectionModule
    elif name == 'PerformanceMonitor':
        from .core import PerformanceMonitor
        return PerformanceMonitor
    elif name == 'MetaController':
        from .meta_controller import MetaController
        return MetaController
    elif name == 'MetaControllerConfig':
        from .meta_controller import MetaControllerConfig
        return MetaControllerConfig
    elif name == 'GradientAnalyzer':
        from .meta_controller import GradientAnalyzer
        return GradientAnalyzer
    elif name == 'DynamicLearningRateScheduler':
        from .meta_controller import DynamicLearningRateScheduler
        return DynamicLearningRateScheduler
    elif name == 'CurriculumStrategy':
        from .meta_controller import CurriculumStrategy
        return CurriculumStrategy
    elif name == 'ProductionAdapter':
        from .production import ProductionAdapter
        return ProductionAdapter
    elif name == 'InferenceMode':
        from .production import InferenceMode
        return InferenceMode
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'AdaptiveFramework',
    'AdaptiveFrameworkConfig',
    'IntrospectionModule',
    'PerformanceMonitor',
    'MetaController',
    'MetaControllerConfig',
    'GradientAnalyzer',
    'DynamicLearningRateScheduler',
    'CurriculumStrategy',
    'ProductionAdapter',
    'InferenceMode'
]
