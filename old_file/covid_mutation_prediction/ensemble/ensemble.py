"""
Ensemble learning methods for COVID-19 mutation prediction.

This module provides various ensemble techniques to combine multiple
models for improved prediction performance.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import logging
from pathlib import Path
import pickle


class EnsembleLearning:
    """
    Ensemble learning class for combining multiple mutation prediction models.
    """
    
    def __init__(self, models: List[nn.Module], 
                 ensemble_method: str = 'average',
                 weights: Optional[List[float]] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize ensemble learning.
        
        Args:
            models: List of trained PyTorch models
            ensemble_method: Method for combining predictions ('average', 'weighted', 'stacking')
            weights: Weights for weighted averaging (if applicable)
            device: Device for computation
        """
        self.models = models
        self.ensemble_method = ensemble_method
        self.weights = weights
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move models to device
        for model in self.models:
            model.to(self.device)
            model.eval()
        
        # Validate weights
        if weights is not None:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            if abs(sum(weights) - 1.0) > 1e-6:
                raise ValueError("Weights must sum to 1.0")
        
        self.logger = logging.getLogger(__name__)
    
    def predict_ensemble(self, data_loader) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            data_loader: DataLoader with input data
            
        Returns:
            Ensemble predictions
        """
        all_predictions = []
        
        # Get predictions from each model
        for i, model in enumerate(self.models):
            model.eval()
            predictions = []
            
            with torch.no_grad():
                for data, _ in data_loader:
                    data = data.to(self.device)
                    
                    outputs = model(data)
                    if isinstance(outputs, dict):
                        output = outputs['predictions']
                    else:
                        output = outputs
                    
                    pred_probs = torch.sigmoid(output.squeeze()).cpu().numpy()
                    predictions.extend(pred_probs)
            
            all_predictions.append(np.array(predictions))
        
        # Combine predictions
        if self.ensemble_method == 'average':
            ensemble_predictions = np.mean(all_predictions, axis=0)
        elif self.ensemble_method == 'weighted':
            if self.weights is None:
                raise ValueError("Weights required for weighted ensemble")
            ensemble_predictions = np.average(all_predictions, axis=0, weights=self.weights)
        elif self.ensemble_method == 'median':
            ensemble_predictions = np.median(all_predictions, axis=0)
        elif self.ensemble_method == 'max':
            ensemble_predictions = np.max(all_predictions, axis=0)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return ensemble_predictions
    
    def evaluate_ensemble(self, data_loader, y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluate ensemble performance.
        
        Args:
            data_loader: DataLoader with test data
            y_true: True labels
            
        Returns:
            Evaluation metrics
        """
        ensemble_predictions = self.predict_ensemble(data_loader)
        
        # Calculate metrics
        auc = roc_auc_score(y_true, ensemble_predictions) if len(set(y_true)) > 1 else 0.0
        ap = average_precision_score(y_true, ensemble_predictions) if len(set(y_true)) > 1 else 0.0
        
        return {
            'ensemble_auc': auc,
            'ensemble_ap': ap,
            'method': self.ensemble_method
        }


class ModelComparison:
    """
    Utility class for comparing multiple models and selecting the best ensemble.
    """
    
    def __init__(self, models: List[nn.Module], model_names: List[str] = None):
        """
        Initialize model comparison.
        
        Args:
            models: List of trained models
            model_names: Names for the models
        """
        self.models = models
        self.model_names = model_names or [f'Model_{i+1}' for i in range(len(models))]
        self.results = {}
    
    def compare_models(self, data_loader, y_true: np.ndarray,
                      device: Optional[torch.device] = None) -> pd.DataFrame:
        """
        Compare individual model performances.
        
        Args:
            data_loader: Test data loader
            y_true: True labels
            device: Device for computation
            
        Returns:
            Comparison DataFrame
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        comparison_data = []
        
        for i, (model, name) in enumerate(zip(self.models, self.model_names)):
            model.to(device)
            model.eval()
            
            predictions = []
            
            with torch.no_grad():
                for data, _ in data_loader:
                    data = data.to(device)
                    
                    outputs = model(data)
                    if isinstance(outputs, dict):
                        output = outputs['predictions']
                    else:
                        output = outputs
                    
                    pred_probs = torch.sigmoid(output.squeeze()).cpu().numpy()
                    predictions.extend(pred_probs)
            
            predictions = np.array(predictions)
            
            # Calculate metrics
            if len(set(y_true)) > 1:
                auc = roc_auc_score(y_true, predictions)
                ap = average_precision_score(y_true, predictions)
            else:
                auc = 0.0
                ap = 0.0
            
            comparison_data.append({
                'model_name': name,
                'auc': auc,
                'average_precision': ap,
                'model_index': i
            })
            
            self.results[name] = predictions
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df.sort_values('auc', ascending=False)
