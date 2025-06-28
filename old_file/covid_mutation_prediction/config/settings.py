"""
Configuration classes for COVID-19 mutation prediction models.

This module contains configuration classes that define the parameters
for model architecture, training, and evaluation.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple


@dataclass
class ModelConfig:
    """Configuration for the mutation transformer model."""
    
    # Model architecture
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_seq_length: int = 1000
    
    # Feature dimensions
    input_dim: int = 9  # Updated to 9 features including freq
    output_dim: int = 1
    
    # Advanced features
    use_positional_encoding: bool = True
    use_layer_norm: bool = True
    use_residual_connections: bool = True
    activation_function: str = "gelu"
    
    # Regularization
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if self.dropout < 0 or self.dropout > 1:
            raise ValueError("dropout must be between 0 and 1")


@dataclass
class TrainingConfig:
    """Configuration for training the mutation prediction model."""
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    patience: int = 10
    min_delta: float = 1e-6
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_steps: int = 1000
    
    # Loss function
    loss_function: str = "combined"  # focal, asymmetric, label_smoothing, combined
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    asymmetric_alpha: float = 0.25
    asymmetric_beta: float = 4.0
    label_smoothing: float = 0.1
    
    # Data handling
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    
    # Advanced training
    use_mixed_precision: bool = True
    accumulate_grad_batches: int = 1
    max_grad_norm: float = 1.0
    
    # Checkpointing
    save_top_k: int = 3
    monitor_metric: str = "val_auc"
    mode: str = "max"
    
    def __post_init__(self):
        """Validate training configuration."""
        if not abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation and metrics."""
    
    # Evaluation metrics
    metrics: List[str] = None
    threshold: float = 0.5
    
    # Cross-validation
    cv_folds: int = 5
    cv_strategy: str = "stratified"  # stratified, time_series, random
    
    # Visualization
    plot_confusion_matrix: bool = True
    plot_roc_curve: bool = True
    plot_precision_recall: bool = True
    plot_feature_importance: bool = True
    
    # Output settings
    save_predictions: bool = True
    save_metrics: bool = True
    output_dir: str = "evaluation_results"
    
    def __post_init__(self):
        """Set default metrics if not provided."""
        if self.metrics is None:
            self.metrics = [
                "accuracy", "precision", "recall", "f1", 
                "auc", "ap", "mcc", "balanced_accuracy"
            ]


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    
    # Optimization settings
    n_trials: int = 100
    timeout: int = 3600  # seconds
    sampler: str = "tpe"  # tpe, random, cmaes
    pruner: str = "median"  # median, hyperband, none
    
    # Search space
    search_spaces: Dict[str, Dict[str, Any]] = None
    
    # Study settings
    direction: str = "maximize"
    study_name: Optional[str] = None
    storage: Optional[str] = None
    
    def __post_init__(self):
        """Set default search spaces if not provided."""
        if self.search_spaces is None:
            self.search_spaces = {
                "learning_rate": {"type": "log_uniform", "low": 1e-5, "high": 1e-2},
                "batch_size": {"type": "categorical", "choices": [16, 32, 64, 128]},
                "d_model": {"type": "categorical", "choices": [128, 256, 512]},
                "num_layers": {"type": "int", "low": 3, "high": 8},
                "dropout": {"type": "uniform", "low": 0.1, "high": 0.5},
                "focal_gamma": {"type": "uniform", "low": 1.0, "high": 3.0}
            }


# Factory functions for creating default configurations
def create_default_model_config() -> ModelConfig:
    """Create a default model configuration."""
    return ModelConfig()


def create_default_training_config() -> TrainingConfig:
    """Create a default training configuration."""
    return TrainingConfig()


def create_default_evaluation_config() -> EvaluationConfig:
    """Create a default evaluation configuration."""
    return EvaluationConfig()


def create_default_optimization_config() -> OptimizationConfig:
    """Create a default optimization configuration."""
    return OptimizationConfig()


def load_config_from_dict(config_dict: Dict[str, Any], config_type: str):
    """Load configuration from dictionary."""
    config_classes = {
        "model": ModelConfig,
        "training": TrainingConfig,
        "evaluation": EvaluationConfig,
        "optimization": OptimizationConfig
    }
    
    if config_type not in config_classes:
        raise ValueError(f"Unknown config type: {config_type}")
    
    return config_classes[config_type](**config_dict)
