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


"""

__version__ = "0.2.5"
__author__ = "Suryaansh Prithvijit Singh"

# Lazy imports to handle circular dependencies
def __getattr__(name):
    if name == 'AdaptiveFramework':
        from .core import AdaptiveFramework
        return AdaptiveFramework
    elif name == 'AdaptiveFrameworkConfig':
        from .core import AdaptiveFrameworkConfig
        return AdaptiveFrameworkConfig
    elif name == 'IntrospectionModule':
        from .core import IntrospectionEngine
        return IntrospectionEngine
    elif name == 'EWCHandler':
        from .ewc import EWCHandler  # Change to .ewc if you created ewc.py
        return EWCHandler
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
    'IntrospectionEngine',
    'PerformanceMonitor',
    'EWCHandler',
    'MetaController',
    'MetaControllerConfig',
    'GradientAnalyzer',
    'DynamicLearningRateScheduler',
    'CurriculumStrategy',
    'ProductionAdapter',
    'InferenceMode'
]
