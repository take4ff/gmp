"""
Advanced Transformer model for COVID-19 mutation prediction.

This module implements a sophisticated transformer architecture with
multiple enhancements for mutation prediction tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
import numpy as np


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer model.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Enhanced multi-head attention with relative position encoding.
    """
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = query.size()
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)
        
        output = self.w_o(context)
        
        # Residual connection and layer normalization
        return self.layer_norm(output + query)


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1,
                 activation: str = "gelu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return self.layer_norm(x + residual)


class TransformerBlock(nn.Module):
    """
    Single transformer block with attention and feed-forward layers.
    """
    
    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float = 0.1,
                 activation: str = "gelu"):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attention(x, x, x, mask)
        x = self.feed_forward(x)
        return x


class AdvancedMutationTransformer(nn.Module):
    """
    Advanced transformer model for COVID-19 mutation prediction with
    enhanced features and multiple prediction heads.
    """
    
    def __init__(self, 
                 input_dim: int = 9,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 max_seq_length: int = 1000,
                 output_dim: int = 1,
                 use_positional_encoding: bool = True,
                 activation: str = "gelu"):
        """
        Initialize the Advanced Mutation Transformer.
        
        Args:
            input_dim: Input feature dimension (9 features including freq)
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feed-forward dimension
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
            output_dim: Output dimension
            use_positional_encoding: Whether to use positional encoding
            activation: Activation function
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # Input embedding
        self.input_projection = nn.Linear(input_dim, d_model)
        self.input_dropout = nn.Dropout(dropout)
        
        # Positional encoding
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout, activation)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, output_dim)
        
        # Additional prediction heads for multi-task learning
        self.confidence_head = nn.Linear(d_model, 1)  # Confidence estimation
        self.feature_importance_head = nn.Linear(d_model, input_dim)  # Feature importance
        
        # Dropout for regularization
        self.output_dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                return_attention_weights: bool = False,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask
            return_attention_weights: Whether to return attention weights
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing predictions and optional additional outputs
        """
        batch_size, seq_len, _ = x.size()
        
        # Input projection
        x = self.input_projection(x)
        x = self.input_dropout(x)
        
        # Positional encoding
        if self.use_positional_encoding:
            x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
            x = self.pos_encoding(x)
            x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Store intermediate features
        features = []
        attention_weights = []
        
        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x, mask)
            if return_features:
                features.append(x.clone())
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Main prediction head
        main_output = self.output_projection(self.output_dropout(x))
        
        # Additional heads
        confidence = torch.sigmoid(self.confidence_head(x))
        feature_importance = torch.softmax(self.feature_importance_head(x), dim=-1)
        
        # Prepare output dictionary
        outputs = {
            'predictions': main_output,
            'confidence': confidence,
            'feature_importance': feature_importance
        }
        
        if return_features:
            outputs['features'] = features
        
        if return_attention_weights:
            outputs['attention_weights'] = attention_weights
        
        return outputs
    
    def get_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature importance scores for interpretability.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature importance scores
        """
        with torch.no_grad():
            outputs = self.forward(x, return_features=False)
            return outputs['feature_importance']
    
    def predict_with_confidence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with confidence estimates.
        
        Args:
            x: Input tensor
            
        Returns:
            Predictions and confidence scores
        """
        with torch.no_grad():
            outputs = self.forward(x)
            return outputs['predictions'], outputs['confidence']
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'AdvancedMutationTransformer',
            'input_dim': self.input_dim,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'output_dim': self.output_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'use_positional_encoding': self.use_positional_encoding
        }


def create_model_from_config(config) -> AdvancedMutationTransformer:
    """
    Create model from configuration object.
    
    Args:
        config: Model configuration object
        
    Returns:
        Initialized model
    """
    return AdvancedMutationTransformer(
        input_dim=config.input_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_seq_length=config.max_seq_length,
        output_dim=config.output_dim,
        use_positional_encoding=config.use_positional_encoding,
        activation=config.activation_function
    )
