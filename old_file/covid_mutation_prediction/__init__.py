"""
COVID-19 Mutation Prediction Package

A modular package for predicting COVID-19 mutations using transformer-based models.
This package provides a complete pipeline for data processing, model training,
evaluation, and ensemble learning.
"""

__version__ = "1.0.0"
__author__ = "COVID-19 Mutation Research Team"

# Import main classes for easy access
try:
    from .config.settings import ModelConfig, TrainingConfig, EvaluationConfig
    from .data.processor import ImprovedDataProcessor
    from .data.dataset import AdvancedMutationDataset
    from .models.transformer import AdvancedMutationTransformer
    from .models.losses import FocalLoss, AsymmetricLoss, LabelSmoothingBCELoss, CombinedLoss
    from .training.pipeline import ImprovedTrainingPipeline
    from .evaluation.metrics import CompositeEvaluator
    from .ensemble.ensemble import EnsembleLearning
    from .optimization.hyperparameter_tuning import HyperparameterOptimizer, OptunaOptimizer
    from .utils.helpers import set_reproducibility, setup_logging
    
    __all__ = [
        'ModelConfig',
        'TrainingConfig', 
        'EvaluationConfig',
        'ImprovedDataProcessor',
        'AdvancedMutationDataset',
        'AdvancedMutationTransformer',
        'FocalLoss',
        'AsymmetricLoss',
        'LabelSmoothingBCELoss',
        'CombinedLoss',
        'ImprovedTrainingPipeline',
        'CompositeEvaluator',
        'EnsembleLearning',
        'HyperparameterOptimizer',
        'OptunaOptimizer',
        'set_reproducibility',
        'setup_logging'
    ]
    
except ImportError as e:
    print(f"Warning: Some modules could not be imported: {e}")
    print("Please ensure all dependencies are installed.")
    
    # Minimal exports if imports fail
    __all__ = []
