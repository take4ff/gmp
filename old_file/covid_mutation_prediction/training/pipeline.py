"""
Advanced training pipeline for COVID-19 mutation prediction models.

This module provides a comprehensive training pipeline with support for
early stopping, learning rate scheduling, model checkpointing, and
advanced optimization techniques.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import time
import os
from pathlib import Path
import json
import pickle
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns


class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6, 
                 mode: str = 'min', restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for minimizing metric, 'max' for maximizing
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.patience_counter = 0
        self.best_weights = None
        self.should_stop = False
        
        self.monitor_op = np.less if mode == 'min' else np.greater
        self.min_delta *= -1 if mode == 'min' else 1
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            model: Model to potentially save weights from
            
        Returns:
            Whether training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.best_weights = model.state_dict().copy()
        elif self.monitor_op(score, self.best_score + self.min_delta):
            self.best_score = score
            self.patience_counter = 0
            self.best_weights = model.state_dict().copy()
        else:
            self.patience_counter += 1
        
        if self.patience_counter >= self.patience:
            self.should_stop = True
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
        
        return self.should_stop


class MetricsTracker:
    """
    Utility class to track training and validation metrics.
    """
    
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_auc': [],
            'val_auc': [],
            'train_ap': [],
            'val_ap': [],
            'learning_rate': []
        }
        self.best_metrics = {}
    
    def update(self, **kwargs):
        """Update metrics with new values."""
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def get_best(self, metric_name: str, mode: str = 'max') -> float:
        """Get best value for a metric."""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        
        values = self.metrics[metric_name]
        return max(values) if mode == 'max' else min(values)
    
    def plot_metrics(self, save_path: Optional[str] = None):
        """Plot training metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.metrics['train_loss'], label='Train Loss', alpha=0.8)
        axes[0, 0].plot(self.metrics['val_loss'], label='Val Loss', alpha=0.8)
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # AUC
        axes[0, 1].plot(self.metrics['train_auc'], label='Train AUC', alpha=0.8)
        axes[0, 1].plot(self.metrics['val_auc'], label='Val AUC', alpha=0.8)
        axes[0, 1].set_title('AUC Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Average Precision
        axes[1, 0].plot(self.metrics['train_ap'], label='Train AP', alpha=0.8)
        axes[1, 0].plot(self.metrics['val_ap'], label='Val AP', alpha=0.8)
        axes[1, 0].set_title('Average Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AP')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 1].plot(self.metrics['learning_rate'], alpha=0.8)
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class ImprovedTrainingPipeline:
    """
    Comprehensive training pipeline for mutation prediction models.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config,
                 device: Optional[torch.device] = None,
                 checkpoint_dir: str = 'checkpoints',
                 experiment_name: str = 'mutation_prediction'):
        """
        Initialize training pipeline.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration object
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            experiment_name: Name of the experiment
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_loss_function()
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Training utilities
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
            mode='max' if config.monitor_metric in ['auc', 'ap', 'accuracy'] else 'min'
        )
        self.metrics_tracker = MetricsTracker()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_map = {
            'adam': optim.Adam,
            'adamw': optim.AdamW,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop
        }
        
        optimizer_class = optimizer_map.get(self.config.optimizer.lower(), optim.AdamW)
        
        if self.config.optimizer.lower() == 'sgd':
            return optimizer_class(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay if hasattr(self.config, 'weight_decay') else 1e-4
            )
        else:
            return optimizer_class(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay if hasattr(self.config, 'weight_decay') else 1e-4
            )
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if self.config.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.num_epochs
            )
        elif self.config.scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.num_epochs // 3,
                gamma=0.1
            )
        elif self.config.scheduler == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max' if self.config.monitor_metric in ['auc', 'ap'] else 'min',
                patience=self.config.patience // 2,
                factor=0.5
            )
        return None
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on configuration."""
        from ..models.losses import get_loss_function
        
        if self.config.loss_function == 'focal':
            return get_loss_function('focal', 
                                   alpha=self.config.focal_alpha,
                                   gamma=self.config.focal_gamma)
        elif self.config.loss_function == 'asymmetric':
            return get_loss_function('asymmetric',
                                   alpha=self.config.asymmetric_alpha,
                                   beta=self.config.asymmetric_beta)
        elif self.config.loss_function == 'combined':
            return get_loss_function('combined',
                                   focal_alpha=self.config.focal_alpha,
                                   focal_gamma=self.config.focal_gamma,
                                   asymmetric_alpha=self.config.asymmetric_alpha,
                                   asymmetric_beta=self.config.asymmetric_beta,
                                   label_smoothing=self.config.label_smoothing)
        else:
            return nn.BCEWithLogitsLoss()
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        predictions = []
        targets = []
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler and self.config.use_mixed_precision:
                with autocast():
                    outputs = self.model(data)
                    if isinstance(outputs, dict):
                        output = outputs['predictions']
                    else:
                        output = outputs
                    loss = self.criterion(output.squeeze(), target.float())
                
                self.scaler.scale(loss).backward()
                
                if self.config.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                 self.config.max_grad_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(data)
                if isinstance(outputs, dict):
                    output = outputs['predictions']
                else:
                    output = outputs
                loss = self.criterion(output.squeeze(), target.float())
                
                loss.backward()
                
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                 self.config.max_grad_norm)
                
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Collect predictions and targets for metrics
            with torch.no_grad():
                pred_probs = torch.sigmoid(output.squeeze()).cpu().numpy()
                predictions.extend(pred_probs)
                targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        auc = roc_auc_score(targets, predictions) if len(set(targets)) > 1 else 0.0
        ap = average_precision_score(targets, predictions) if len(set(targets)) > 1 else 0.0
        
        return {
            'loss': avg_loss,
            'auc': auc,
            'ap': ap
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                outputs = self.model(data)
                if isinstance(outputs, dict):
                    output = outputs['predictions']
                else:
                    output = outputs
                
                loss = self.criterion(output.squeeze(), target.float())
                total_loss += loss.item()
                
                # Collect predictions and targets
                pred_probs = torch.sigmoid(output.squeeze()).cpu().numpy()
                predictions.extend(pred_probs)
                targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        auc = roc_auc_score(targets, predictions) if len(set(targets)) > 1 else 0.0
        ap = average_precision_score(targets, predictions) if len(set(targets)) > 1 else 0.0
        
        return {
            'loss': avg_loss,
            'auc': auc,
            'ap': ap
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], 
                       is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else str(self.config)
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'{self.experiment_name}_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / f'{self.experiment_name}_best.pth'
            torch.save(checkpoint, best_path)
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Training history and final metrics
        """
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        best_metric = 0.0 if self.config.monitor_metric in ['auc', 'ap'] else float('inf')
        
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics[self.config.monitor_metric])
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update metrics tracker
            self.metrics_tracker.update(
                train_loss=train_metrics['loss'],
                val_loss=val_metrics['loss'],
                train_auc=train_metrics['auc'],
                val_auc=val_metrics['auc'],
                train_ap=train_metrics['ap'],
                val_ap=val_metrics['ap'],
                learning_rate=current_lr
            )
            
            # Check for best model
            current_metric = val_metrics[self.config.monitor_metric]
            is_best = False
            if self.config.monitor_metric in ['auc', 'ap']:
                if current_metric > best_metric:
                    best_metric = current_metric
                    is_best = True
            else:
                if current_metric < best_metric:
                    best_metric = current_metric
                    is_best = True
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping check
            if self.early_stopping(current_metric, self.model):
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break
            
            # Logging
            epoch_time = time.time() - epoch_start
            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val AUC: {val_metrics['auc']:.4f}, "
                f"Val AP: {val_metrics['ap']:.4f}, "
                f"LR: {current_lr:.2e}, "
                f"Time: {epoch_time:.2f}s"
            )
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")
        
        # Plot metrics
        self.metrics_tracker.plot_metrics(
            save_path=self.checkpoint_dir / f'{self.experiment_name}_metrics.png'
        )
        
        # Save training history
        history_path = self.checkpoint_dir / f'{self.experiment_name}_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.metrics_tracker.metrics, f, indent=2)
        
        return {
            'metrics': self.metrics_tracker.metrics,
            'best_metrics': {
                'best_val_auc': self.metrics_tracker.get_best('val_auc', 'max'),
                'best_val_ap': self.metrics_tracker.get_best('val_ap', 'max'),
                'best_val_loss': self.metrics_tracker.get_best('val_loss', 'min')
            },
            'total_time': total_time,
            'final_epoch': len(self.metrics_tracker.metrics['train_loss'])
        }
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        return checkpoint.get('metrics', {})
