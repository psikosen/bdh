# Dragon Hatching Training Package
# Multi-task training for text coherence, function calling, tool use, and bash

from .data_generators import (
    UnifiedDataGenerator,
    FunctionCallGenerator,
    BashCommandGenerator,
    CoherenceGenerator,
    ToolUseGenerator,
    ChainOfThoughtGenerator,
    TaskType,
    TrainingSample,
)

from .evaluation import (
    UnifiedEvaluator,
    CoherenceEvaluator,
    FunctionCallEvaluator,
    BashEvaluator,
    ToolUseEvaluator,
    CoherenceMetrics,
    FunctionCallMetrics,
    BashMetrics,
    ToolUseMetrics,
)

from .train_multitask import (
    TrainingConfig,
    MultiTaskDataset,
    MultiTaskTrainer,
)

__all__ = [
    # Generators
    'UnifiedDataGenerator',
    'FunctionCallGenerator',
    'BashCommandGenerator',
    'CoherenceGenerator',
    'ToolUseGenerator',
    'ChainOfThoughtGenerator',
    'TaskType',
    'TrainingSample',

    # Evaluators
    'UnifiedEvaluator',
    'CoherenceEvaluator',
    'FunctionCallEvaluator',
    'BashEvaluator',
    'ToolUseEvaluator',

    # Metrics
    'CoherenceMetrics',
    'FunctionCallMetrics',
    'BashMetrics',
    'ToolUseMetrics',

    # Training
    'TrainingConfig',
    'MultiTaskDataset',
    'MultiTaskTrainer',
]
