"""
Advanced loss functions for COVID-19 mutation prediction.

This module implements various loss functions optimized for imbalanced
binary classification tasks in mutation prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in binary classification.
    
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    where p_t is the predicted probability for the ground truth class.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, 
                 reduce: str = 'mean', eps: float = 1e-8):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (typically between 0.25-0.75)
            gamma: Focusing parameter (typically 2.0)
            reduce: Reduction method ('mean', 'sum', 'none')
            eps: Small epsilon for numerical stability
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
        self.eps = eps
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Focal Loss.
        
        Args:
            inputs: Predicted logits or probabilities
            targets: Ground truth binary labels
            
        Returns:
            Computed focal loss
        """
        # Ensure inputs are probabilities
        if inputs.dim() > 1 and inputs.size(1) > 1:
            probs = F.softmax(inputs, dim=1)[:, 1]  # Get positive class probability
        else:
            probs = torch.sigmoid(inputs.squeeze())
        
        # Clip probabilities for numerical stability
        probs = torch.clamp(probs, self.eps, 1 - self.eps)
        
        # Compute focal loss
        ce_loss = F.binary_cross_entropy(probs, targets.float(), reduction='none')
        
        # Compute p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Compute alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply reduction
        if self.reduce == 'mean':
            return focal_loss.mean()
        elif self.reduce == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for imbalanced classification.
    
    This loss applies different focusing parameters for positive and negative samples.
    """
    
    def __init__(self, alpha: float = 0.25, beta: float = 4.0, 
                 reduce: str = 'mean', eps: float = 1e-8):
        """
        Initialize Asymmetric Loss.
        
        Args:
            alpha: Weighting factor for positive class
            beta: Asymmetric focusing parameter
            reduce: Reduction method
            eps: Small epsilon for numerical stability
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduce = reduce
        self.eps = eps
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Asymmetric Loss.
        
        Args:
            inputs: Predicted logits or probabilities
            targets: Ground truth binary labels
            
        Returns:
            Computed asymmetric loss
        """
        # Ensure inputs are probabilities
        if inputs.dim() > 1 and inputs.size(1) > 1:
            probs = F.softmax(inputs, dim=1)[:, 1]
        else:
            probs = torch.sigmoid(inputs.squeeze())
        
        # Clip probabilities
        probs = torch.clamp(probs, self.eps, 1 - self.eps)
        
        # Positive and negative components
        targets_float = targets.float()
        
        # Positive loss (focus on hard positive examples)
        pos_loss = targets_float * (1 - probs) ** self.beta * torch.log(probs)
        
        # Negative loss (standard for negative examples)
        neg_loss = (1 - targets_float) * probs ** self.alpha * torch.log(1 - probs)
        
        # Combine losses
        loss = -(pos_loss + neg_loss)
        
        # Apply reduction
        if self.reduce == 'mean':
            return loss.mean()
        elif self.reduce == 'sum':
            return loss.sum()
        else:
            return loss


class LabelSmoothingBCELoss(nn.Module):
    """
    Binary Cross-Entropy Loss with Label Smoothing.
    
    This helps prevent overfitting and improves calibration.
    """
    
    def __init__(self, smoothing: float = 0.1, reduce: str = 'mean'):
        """
        Initialize Label Smoothing BCE Loss.
        
        Args:
            smoothing: Label smoothing factor (0 = no smoothing)
            reduce: Reduction method
        """
        super().__init__()
        self.smoothing = smoothing
        self.reduce = reduce
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Label Smoothing BCE Loss.
        
        Args:
            inputs: Predicted logits or probabilities
            targets: Ground truth binary labels
            
        Returns:
            Computed label smoothing BCE loss
        """
        # Ensure inputs are probabilities
        if inputs.dim() > 1 and inputs.size(1) > 1:
            probs = F.softmax(inputs, dim=1)[:, 1]
        else:
            probs = torch.sigmoid(inputs.squeeze())
        
        # Apply label smoothing
        targets_smooth = targets.float() * (1 - self.smoothing) + 0.5 * self.smoothing
        
        # Compute BCE loss
        loss = F.binary_cross_entropy(probs, targets_smooth, reduction=self.reduce)
        
        return loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss for handling class imbalance.
    """
    
    def __init__(self, pos_weight: Optional[float] = None, reduce: str = 'mean'):
        """
        Initialize Weighted BCE Loss.
        
        Args:
            pos_weight: Weight for positive class (calculated automatically if None)
            reduce: Reduction method
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.reduce = reduce
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Weighted BCE Loss.
        
        Args:
            inputs: Predicted logits or probabilities
            targets: Ground truth binary labels
            
        Returns:
            Computed weighted BCE loss
        """
        # Calculate pos_weight if not provided
        if self.pos_weight is None:
            pos_count = targets.sum().item()
            neg_count = len(targets) - pos_count
            if pos_count > 0:
                pos_weight = neg_count / pos_count
            else:
                pos_weight = 1.0
        else:
            pos_weight = self.pos_weight
        
        # Convert to tensor
        pos_weight_tensor = torch.tensor(pos_weight, device=inputs.device)
        
        # Compute weighted BCE loss
        loss = F.binary_cross_entropy_with_logits(
            inputs.squeeze(), targets.float(), 
            pos_weight=pos_weight_tensor, reduction=self.reduce
        )
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function that incorporates multiple loss components.
    """
    
    def __init__(self, 
                 focal_weight: float = 0.5,
                 asymmetric_weight: float = 0.3,
                 bce_weight: float = 0.2,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 asymmetric_alpha: float = 0.25,
                 asymmetric_beta: float = 4.0,
                 label_smoothing: float = 0.1):
        """
        Initialize Combined Loss.
        
        Args:
            focal_weight: Weight for focal loss component
            asymmetric_weight: Weight for asymmetric loss component
            bce_weight: Weight for BCE loss component
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            asymmetric_alpha: Alpha parameter for asymmetric loss
            asymmetric_beta: Beta parameter for asymmetric loss
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        
        # Ensure weights sum to 1
        total_weight = focal_weight + asymmetric_weight + bce_weight
        self.focal_weight = focal_weight / total_weight
        self.asymmetric_weight = asymmetric_weight / total_weight
        self.bce_weight = bce_weight / total_weight
        
        # Initialize loss functions
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.asymmetric_loss = AsymmetricLoss(alpha=asymmetric_alpha, beta=asymmetric_beta)
        self.bce_loss = LabelSmoothingBCELoss(smoothing=label_smoothing)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Combined Loss.
        
        Args:
            inputs: Predicted logits or probabilities
            targets: Ground truth binary labels
            
        Returns:
            Computed combined loss
        """
        # Compute individual losses
        focal = self.focal_loss(inputs, targets)
        asymmetric = self.asymmetric_loss(inputs, targets)
        bce = self.bce_loss(inputs, targets)
        
        # Combine losses
        combined = (self.focal_weight * focal + 
                   self.asymmetric_weight * asymmetric + 
                   self.bce_weight * bce)
        
        return combined


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation-like tasks.
    """
    
    def __init__(self, smooth: float = 1e-6):
        """
        Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Dice Loss.
        
        Args:
            inputs: Predicted logits or probabilities
            targets: Ground truth binary labels
            
        Returns:
            Computed dice loss
        """
        # Ensure inputs are probabilities
        if inputs.dim() > 1 and inputs.size(1) > 1:
            probs = F.softmax(inputs, dim=1)[:, 1]
        else:
            probs = torch.sigmoid(inputs.squeeze())
        
        # Flatten tensors
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1).float()
        
        # Compute Dice coefficient
        intersection = (probs_flat * targets_flat).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        
        # Dice loss = 1 - Dice coefficient
        return 1 - dice_coeff


def get_loss_function(loss_name: str, **kwargs) -> nn.Module:
    """
    Factory function to get loss function by name.
    
    Args:
        loss_name: Name of the loss function
        **kwargs: Additional arguments for loss function
        
    Returns:
        Initialized loss function
    """
    loss_functions = {
        'bce': nn.BCEWithLogitsLoss,
        'focal': FocalLoss,
        'asymmetric': AsymmetricLoss,
        'label_smoothing': LabelSmoothingBCELoss,
        'weighted_bce': WeightedBCELoss,
        'combined': CombinedLoss,
        'dice': DiceLoss
    }
    
    if loss_name not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_name}")
    
    return loss_functions[loss_name](**kwargs)
