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

__version__ = "0.2.5.6"
__license__ = "MIT"
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
    elif name == 'IntrospectionEngine':
        from .core import IntrospectionEngine
        return IntrospectionEngine
    elif name == 'EWCHandler':
        from .ewc import EWCHandler
        return EWCHandler
    elif name == 'PerformanceMonitor':
        from .core import PerformanceMonitor
        return PerformanceMonitor
    elif name == 'PerformanceSnapshot':
        from .core import PerformanceSnapshot
        return PerformanceSnapshot
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
    elif name == 'UnifiedMemoryHandler':
        from .memory import UnifiedMemoryHandler
        return UnifiedMemoryHandler
    elif name == 'PrioritizedReplayBuffer':
        from .memory import PrioritizedReplayBuffer
        return PrioritizedReplayBuffer
    elif name == 'AdaptiveRegularization':
        from .memory import AdaptiveRegularization
        return AdaptiveRegularization
    elif name == 'DynamicConsolidationScheduler':
        from .memory import DynamicConsolidationScheduler
        return DynamicConsolidationScheduler
    elif name == 'SIHandler':
        from .ewc import SIHandler
        return SIHandler
    elif name == 'ConsciousnessCore':
        from .consciousness import ConsciousnessCore
        return ConsciousnessCore
    elif name == 'AttentionMechanism':
        from .consciousness import AttentionMechanism
        return AttentionMechanism
    elif name == 'IntrinisicMotivation':
        from .consciousness import IntrinisicMotivation
        return IntrinisicMotivation
    elif name == 'SelfAwarenessMonitor':
        from .consciousness import SelfAwarenessMonitor
        return SelfAwarenessMonitor
    elif name == 'ConfigValidator':
        from .validation import ConfigValidator, validate_config
        return ConfigValidator
    elif name == 'validate_config':
        from .validation import validate_config
        return validate_config
    # === V2.0: State-of-the-art Self-Awareness Framework ===
    elif name == 'HumanLikeSelfAwarenessWrapper':
        from .self_awareness_v2 import HumanLikeSelfAwarenessWrapper
        return HumanLikeSelfAwarenessWrapper
    elif name == 'MetaCognitiveAwarenessEngine':
        from .self_awareness_v2 import MetaCognitiveAwarenessEngine
        return MetaCognitiveAwarenessEngine
    elif name == 'MetaCognitiveState':
        from .self_awareness_v2 import MetaCognitiveState
        return MetaCognitiveState
    elif name == 'ConfidenceSignal':
        from .self_awareness_v2 import ConfidenceSignal
        return ConfidenceSignal
    elif name == 'CompetenceSignal':
        from .self_awareness_v2 import CompetenceSignal
        return CompetenceSignal
    elif name == 'AdaptiveLearningController':
        from .self_awareness_v2 import AdaptiveLearningController
        return AdaptiveLearningController
    elif name == 'SelfImprovementPlanner':
        from .self_awareness_v2 import SelfImprovementPlanner
        return SelfImprovementPlanner
    elif name == 'AdaptiveAttentionMechanism':
        from .self_awareness_v2 import AdaptiveAttentionMechanism
        return AdaptiveAttentionMechanism
    elif name == 'OutOfDistributionDetector':
        from .self_awareness_v2 import OutOfDistributionDetector
        return OutOfDistributionDetector
    elif name == 'MirrorMindWithSelfAwareness':
        from .integration_guide import MirrorMindWithSelfAwareness
        return MirrorMindWithSelfAwareness
    elif name == 'MultiTaskSelfAwareLearner':
        from .integration_guide import MultiTaskSelfAwareLearner
        return MultiTaskSelfAwareLearner
    # === PRESETS SYSTEM ===
    elif name == 'PRESETS':
        from .presets import PRESETS
        return PRESETS
    elif name == 'Preset':
        from .presets import Preset
        return Preset
    elif name == 'PresetManager':
        from .presets import PresetManager
        return PresetManager
    elif name == 'load_preset':
        from .presets import load_preset
        return load_preset
    elif name == 'list_presets':
        from .presets import list_presets
        return list_presets
    elif name == 'compare_presets':
        from .presets import compare_presets
        return compare_presets
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'AdaptiveFramework',
    'AdaptiveFrameworkConfig',
    'IntrospectionEngine',
    'PerformanceMonitor',
    'EWCHandler',
    'SIHandler',
    'MetaController',
    'MetaControllerConfig',
    'GradientAnalyzer',
    'DynamicLearningRateScheduler',
    'CurriculumStrategy',
    'ProductionAdapter',
    'InferenceMode',
    # SOTA V7.0 Memory System
    'UnifiedMemoryHandler',
    'PrioritizedReplayBuffer',
    'AdaptiveRegularization',
    'DynamicConsolidationScheduler',
    # V7.0 Consciousness Layer
    'ConsciousnessCore',
    'AttentionMechanism',
    'IntrinisicMotivation',
    'SelfAwarenessMonitor',
    # === V2.0: State-of-the-art Self-Awareness Framework ===
    'HumanLikeSelfAwarenessWrapper',
    'MetaCognitiveAwarenessEngine',
    'MetaCognitiveState',
    'ConfidenceSignal',
    'CompetenceSignal',
    'AdaptiveLearningController',
    'SelfImprovementPlanner',
    'AdaptiveAttentionMechanism',
    'OutOfDistributionDetector',
    'MirrorMindWithSelfAwareness',
    'MultiTaskSelfAwareLearner',
    # === PRESETS SYSTEM ===
    'PRESETS',
    'Preset',
    'PresetManager',
    'load_preset',
    'list_presets',
    'compare_presets',
]
