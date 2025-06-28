"""
Utility functions for COVID-19 mutation prediction package.

This module provides helper functions for data handling, model utilities,
logging, and other common operations.
"""

import torch
import numpy as np
import pandas as pd
import logging
import random
import os
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import yaml
import json
import pickle


def set_reproducibility(seed: int = 42, deterministic: bool = True):
    """
    Set seeds for reproducible results.
    
    Args:
        seed: Random seed value
        deterministic: Whether to enable deterministic operations
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger('covid_mutation_prediction')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def get_device(device_id: Optional[int] = None) -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device_id: Specific GPU device ID (None for auto-selection)
        
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        if device_id is not None:
            device = torch.device(f'cuda:{device_id}')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    return device


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def save_model(model: torch.nn.Module, save_path: str, 
               additional_info: Optional[Dict[str, Any]] = None):
    """
    Save model with additional information.
    
    Args:
        model: PyTorch model to save
        save_path: Path to save the model
        additional_info: Additional information to save
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare save dictionary
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'parameter_count': count_parameters(model)
    }
    
    if additional_info:
        save_dict.update(additional_info)
    
    torch.save(save_dict, save_path)


def load_model(model: torch.nn.Module, load_path: str, 
               device: Optional[torch.device] = None) -> Dict[str, Any]:
    """
    Load model from saved checkpoint.
    
    Args:
        model: PyTorch model instance
        load_path: Path to load the model from
        device: Device to load the model on
        
    Returns:
        Dictionary with loaded information
    """
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return {
        'model_class': checkpoint.get('model_class'),
        'parameter_count': checkpoint.get('parameter_count'),
        'additional_info': {k: v for k, v in checkpoint.items() 
                          if k not in ['model_state_dict', 'model_class', 'parameter_count']}
    }


def calculate_class_weights(y: np.ndarray) -> torch.Tensor:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        y: Target labels
        
    Returns:
        Class weights tensor
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    unique_classes = np.unique(y)
    class_weights = compute_class_weight(
        'balanced', classes=unique_classes, y=y
    )
    
    return torch.FloatTensor(class_weights)


def create_data_splits(data: pd.DataFrame, target_col: str,
                      train_ratio: float = 0.7, val_ratio: float = 0.15,
                      test_ratio: float = 0.15, stratify: bool = True,
                      random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/validation/test splits from data.
    
    Args:
        data: Input DataFrame
        target_col: Name of target column
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        stratify: Whether to stratify splits
        random_state: Random seed
        
    Returns:
        Train, validation, and test DataFrames
    """
    from sklearn.model_selection import train_test_split
    
    # Validate ratios
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    # First split: train vs temp
    temp_ratio = val_ratio + test_ratio
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    if stratify and len(y.unique()) > 1:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=temp_ratio, stratify=y, random_state=random_state
        )
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=temp_ratio, random_state=random_state
        )
    
    # Second split: val vs test
    test_ratio_adjusted = test_ratio / temp_ratio
    
    if stratify and len(y_temp.unique()) > 1:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_ratio_adjusted, 
            stratify=y_temp, random_state=random_state
        )
    else:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_ratio_adjusted, random_state=random_state
        )
    
    # Reconstruct DataFrames
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    return train_df, val_df, test_df


def memory_usage_check():
    """
    Check current memory usage (if GPU is available).
    
    Returns:
        Memory usage information
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        return {
            'allocated_gb': allocated,
            'cached_gb': cached,
            'max_allocated_gb': max_allocated,
            'device_name': torch.cuda.get_device_name()
        }
    else:
        return {'message': 'CUDA not available'}


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate_data_format(data: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that data has required format.
    
    Args:
        data: Input DataFrame
        required_columns: List of required column names
        
    Returns:
        Whether data format is valid
    """
    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return True


def encode_categorical_features(data: pd.DataFrame, categorical_columns: List[str],
                               method: str = 'label') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Encode categorical features.
    
    Args:
        data: Input DataFrame
        categorical_columns: List of categorical column names
        method: Encoding method ('label', 'onehot')
        
    Returns:
        Encoded DataFrame and encoding information
    """
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    
    data_encoded = data.copy()
    encoding_info = {}
    
    for col in categorical_columns:
        if col in data.columns:
            if method == 'label':
                encoder = LabelEncoder()
                data_encoded[col] = encoder.fit_transform(data[col].astype(str))
                encoding_info[col] = {
                    'encoder': encoder,
                    'classes': encoder.classes_.tolist()
                }
            elif method == 'onehot':
                encoder = OneHotEncoder(sparse=False, drop='first')
                encoded_cols = encoder.fit_transform(data[[col]])
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0][1:]]
                
                # Remove original column and add encoded columns
                data_encoded = data_encoded.drop(columns=[col])
                for i, name in enumerate(feature_names):
                    data_encoded[name] = encoded_cols[:, i]
                
                encoding_info[col] = {
                    'encoder': encoder,
                    'feature_names': feature_names,
                    'categories': encoder.categories_[0].tolist()
                }
    
    return data_encoded, encoding_info


def create_feature_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary of dataset features.
    
    Args:
        data: Input DataFrame
        
    Returns:
        Feature summary DataFrame
    """
    summary = pd.DataFrame({
        'column': data.columns,
        'dtype': data.dtypes,
        'null_count': data.isnull().sum(),
        'null_percentage': (data.isnull().sum() / len(data)) * 100,
        'unique_count': data.nunique(),
        'sample_values': [data[col].dropna().head(3).tolist() for col in data.columns]
    })
    
    # Add statistical info for numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        idx = summary[summary['column'] == col].index[0]
        summary.loc[idx, 'mean'] = data[col].mean()
        summary.loc[idx, 'std'] = data[col].std()
        summary.loc[idx, 'min'] = data[col].min()
        summary.loc[idx, 'max'] = data[col].max()
    
    return summary


class Timer:
    """Simple timer context manager."""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f"{self.description} completed in {elapsed:.2f} seconds")
    
    @property
    def elapsed(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"
