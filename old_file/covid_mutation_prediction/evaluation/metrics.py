"""
Comprehensive evaluation metrics and visualization for COVID-19 mutation prediction.

This module provides extensive evaluation capabilities including multiple
metrics, visualizations, and statistical analysis tools.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    matthews_corrcoef, balanced_accuracy_score
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import json


class CompositeEvaluator:
    """
    Comprehensive evaluator for mutation prediction models with
    multiple metrics and visualization capabilities.
    """
    
    def __init__(self, threshold: float = 0.5, save_plots: bool = True,
                 output_dir: str = 'evaluation_results'):
        """
        Initialize the evaluator.
        
        Args:
            threshold: Decision threshold for binary classification
            save_plots: Whether to save generated plots
            output_dir: Directory to save evaluation results
        """
        self.threshold = threshold
        self.save_plots = save_plots
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage
        self.metrics_history = []
        self.predictions_history = []
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        
        # Confusion matrix derived metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        
        # Specificity and sensitivity
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Positive and negative predictive values
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Precision
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        # Likelihood ratios
        sensitivity = metrics['sensitivity']
        specificity = metrics['specificity']
        metrics['positive_lr'] = sensitivity / (1 - specificity) if specificity < 1 else float('inf')
        metrics['negative_lr'] = (1 - sensitivity) / specificity if specificity > 0 else float('inf')
        
        # Probability-based metrics (if available)
        if y_pred_proba is not None:
            if len(np.unique(y_true)) > 1:  # Check if both classes are present
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
                metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
            else:
                metrics['auc'] = 0.0
                metrics['average_precision'] = 0.0
            
            # Brier score (calibration metric)
            metrics['brier_score'] = np.mean((y_pred_proba - y_true) ** 2)
            
            # Log loss
            eps = 1e-15
            y_pred_proba_clipped = np.clip(y_pred_proba, eps, 1 - eps)
            metrics['log_loss'] = -np.mean(y_true * np.log(y_pred_proba_clipped) + 
                                         (1 - y_true) * np.log(1 - y_pred_proba_clipped))
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             title: str = "Confusion Matrix", normalize: bool = False,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            normalize: Whether to normalize the matrix
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        if save_path or self.save_plots:
            if save_path is None:
                save_path = self.output_dir / 'confusion_matrix.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                       title: str = "ROC Curve", save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if len(np.unique(y_true)) <= 1:
            self.logger.warning("Cannot plot ROC curve: only one class present")
            return None
        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        if save_path or self.save_plots:
            if save_path is None:
                save_path = self.output_dir / 'roc_curve.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   title: str = "Precision-Recall Curve",
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot precision-recall curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if len(np.unique(y_true)) <= 1:
            self.logger.warning("Cannot plot PR curve: only one class present")
            return None
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        ap = average_precision_score(y_true, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {ap:.3f})')
        
        # Baseline (random classifier)
        baseline = np.sum(y_true) / len(y_true)
        ax.axhline(y=baseline, color='red', linestyle='--', alpha=0.8,
                   label=f'Baseline (AP = {baseline:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path or self.save_plots:
            if save_path is None:
                save_path = self.output_dir / 'precision_recall_curve.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                              n_bins: int = 10, title: str = "Calibration Curve",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot calibration curve to assess probability calibration.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            n_bins: Number of bins for calibration
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if len(np.unique(y_true)) <= 1:
            self.logger.warning("Cannot plot calibration curve: only one class present")
            return None
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins
        )
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(mean_predicted_value, fraction_of_positives, "s-",
                label="Model", linewidth=2, markersize=8)
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated", alpha=0.8)
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path or self.save_plots:
            if save_path is None:
                save_path = self.output_dir / 'calibration_curve.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_threshold_analysis(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                               title: str = "Threshold Analysis",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot various metrics vs threshold to help choose optimal threshold.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        thresholds = np.linspace(0, 1, 101)
        metrics_vs_threshold = {
            'precision': [],
            'recall': [],
            'f1': [],
            'accuracy': [],
            'balanced_accuracy': []
        }
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            metrics_vs_threshold['precision'].append(
                precision_score(y_true, y_pred, zero_division=0)
            )
            metrics_vs_threshold['recall'].append(
                recall_score(y_true, y_pred, zero_division=0)
            )
            metrics_vs_threshold['f1'].append(
                f1_score(y_true, y_pred, zero_division=0)
            )
            metrics_vs_threshold['accuracy'].append(
                accuracy_score(y_true, y_pred)
            )
            metrics_vs_threshold['balanced_accuracy'].append(
                balanced_accuracy_score(y_true, y_pred)
            )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for metric_name, values in metrics_vs_threshold.items():
            ax.plot(thresholds, values, label=metric_name.capitalize(), linewidth=2)
        
        # Mark current threshold
        ax.axvline(x=self.threshold, color='red', linestyle='--', alpha=0.7,
                   label=f'Current threshold ({self.threshold})')
        
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Metric Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        if save_path or self.save_plots:
            if save_path is None:
                save_path = self.output_dir / 'threshold_analysis.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def evaluate_model(self, model: nn.Module, data_loader, device: torch.device,
                      return_predictions: bool = False) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            model: PyTorch model to evaluate
            data_loader: DataLoader with test data
            device: Device to run evaluation on
            return_predictions: Whether to return predictions
            
        Returns:
            Dictionary containing metrics and optional predictions
        """
        model.eval()
        all_predictions = []
        all_probabilities = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                
                outputs = model(data)
                if isinstance(outputs, dict):
                    output = outputs['predictions']
                else:
                    output = outputs
                
                probabilities = torch.sigmoid(output.squeeze()).cpu().numpy()
                predictions = (probabilities >= self.threshold).astype(int)
                
                all_predictions.extend(predictions)
                all_probabilities.extend(probabilities)
                all_targets.extend(target.cpu().numpy())
        
        # Convert to numpy arrays
        y_true = np.array(all_targets)
        y_pred = np.array(all_predictions)
        y_pred_proba = np.array(all_probabilities)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_pred_proba)
        
        # Generate visualizations
        self.plot_confusion_matrix(y_true, y_pred)
        if len(np.unique(y_true)) > 1:
            self.plot_roc_curve(y_true, y_pred_proba)
            self.plot_precision_recall_curve(y_true, y_pred_proba)
            self.plot_calibration_curve(y_true, y_pred_proba)
            self.plot_threshold_analysis(y_true, y_pred_proba)
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_path = self.output_dir / 'evaluation_metrics.csv'
        metrics_df.to_csv(metrics_path, index=False)
        
        # Save detailed metrics as JSON
        metrics_json_path = self.output_dir / 'evaluation_metrics.json'
        with open(metrics_json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        result = {'metrics': metrics}
        
        if return_predictions:
            result['predictions'] = {
                'y_true': y_true,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        
        return result
    
    def compare_models(self, results: List[Dict[str, Any]], 
                      model_names: List[str] = None) -> pd.DataFrame:
        """
        Compare multiple model evaluation results.
        
        Args:
            results: List of evaluation results
            model_names: Names for the models
            
        Returns:
            DataFrame comparing model metrics
        """
        if model_names is None:
            model_names = [f'Model_{i+1}' for i in range(len(results))]
        
        comparison_data = []
        for i, result in enumerate(results):
            metrics = result['metrics'].copy()
            metrics['model_name'] = model_names[i]
            comparison_data.append(metrics)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Reorder columns
        cols = ['model_name'] + [col for col in comparison_df.columns if col != 'model_name']
        comparison_df = comparison_df[cols]
        
        # Save comparison
        comparison_path = self.output_dir / 'model_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        
        return comparison_df
    
    def generate_report(self, metrics: Dict[str, float], 
                       model_name: str = "Mutation Prediction Model") -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            metrics: Dictionary of computed metrics
            model_name: Name of the model
            
        Returns:
            Formatted report string
        """
        report = f"""
# {model_name} - Evaluation Report

## Classification Metrics
- **Accuracy**: {metrics.get('accuracy', 0):.4f}
- **Balanced Accuracy**: {metrics.get('balanced_accuracy', 0):.4f}
- **Precision**: {metrics.get('precision', 0):.4f}
- **Recall (Sensitivity)**: {metrics.get('recall', 0):.4f}
- **Specificity**: {metrics.get('specificity', 0):.4f}
- **F1-Score**: {metrics.get('f1', 0):.4f}
- **Matthews Correlation Coefficient**: {metrics.get('mcc', 0):.4f}

## Probabilistic Metrics
- **AUC-ROC**: {metrics.get('auc', 0):.4f}
- **Average Precision**: {metrics.get('average_precision', 0):.4f}
- **Brier Score**: {metrics.get('brier_score', 0):.4f}
- **Log Loss**: {metrics.get('log_loss', 0):.4f}

## Confusion Matrix
- **True Positives**: {metrics.get('true_positives', 0)}
- **True Negatives**: {metrics.get('true_negatives', 0)}
- **False Positives**: {metrics.get('false_positives', 0)}
- **False Negatives**: {metrics.get('false_negatives', 0)}

## Predictive Values
- **Positive Predictive Value (PPV)**: {metrics.get('ppv', 0):.4f}
- **Negative Predictive Value (NPV)**: {metrics.get('npv', 0):.4f}

## Likelihood Ratios
- **Positive Likelihood Ratio**: {metrics.get('positive_lr', 0):.4f}
- **Negative Likelihood Ratio**: {metrics.get('negative_lr', 0):.4f}

## Decision Threshold
- **Threshold Used**: {self.threshold}

---
Report generated by CompositeEvaluator
        """
        
        # Save report
        report_path = self.output_dir / 'evaluation_report.md'
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report
