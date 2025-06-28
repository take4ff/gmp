"""
Advanced data processor for COVID-19 mutation prediction.

This module provides comprehensive data processing capabilities including
feature engineering, normalization, sequence handling, and batch preparation.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Any, Optional, Union
import logging
from abc import ABC, abstractmethod


class ImprovedDataProcessor:
    """
    Advanced data processor for mutation prediction with comprehensive
    feature engineering and preprocessing capabilities.
    """
    
    def __init__(self, 
                 scaling_method: str = 'standard',
                 sequence_length: int = 1000,
                 feature_engineering: bool = True,
                 handle_missing: str = 'impute',
                 validation_split: float = 0.2,
                 random_state: int = 42):
        """
        Initialize the data processor.
        
        Args:
            scaling_method: Method for feature scaling ('standard', 'minmax', 'robust', 'none')
            sequence_length: Maximum sequence length for padding/truncation
            feature_engineering: Whether to apply advanced feature engineering
            handle_missing: Method for handling missing values ('impute', 'drop', 'zero')
            validation_split: Fraction of data to use for validation
            random_state: Random seed for reproducibility
        """
        self.scaling_method = scaling_method
        self.sequence_length = sequence_length
        self.feature_engineering = feature_engineering
        self.handle_missing = handle_missing
        self.validation_split = validation_split
        self.random_state = random_state
        
        # Initialize scalers
        self.scaler = self._get_scaler()
        self.fitted = False
        
        # Feature statistics
        self.feature_stats = {}
        self.feature_names = []
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
    def _get_scaler(self):
        """Get the appropriate scaler based on scaling method."""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'none': None
        }
        return scalers.get(self.scaling_method)
    
    def _handle_missing_values(self, data: np.ndarray) -> np.ndarray:
        """Handle missing values in the data."""
        if self.handle_missing == 'drop':
            # Remove rows with any missing values
            mask = ~np.isnan(data).any(axis=1)
            return data[mask]
        elif self.handle_missing == 'zero':
            # Replace missing values with zeros
            return np.nan_to_num(data, nan=0.0)
        elif self.handle_missing == 'impute':
            # Replace missing values with column means
            col_means = np.nanmean(data, axis=0)
            inds = np.where(np.isnan(data))
            data[inds] = np.take(col_means, inds[1])
            return data
        else:
            return data
    
    def _engineer_features(self, features: np.ndarray) -> np.ndarray:
        """Apply advanced feature engineering."""
        if not self.feature_engineering:
            return features
        
        # Original features: [pos, ref_codon_idx, alt_codon_idx, ref_aa_idx, alt_aa_idx, 
        #                    syn_nonsyn, transition_transversion, cpg_site, freq]
        
        engineered_features = []
        
        # Original features
        engineered_features.append(features)
        
        # Position-based features
        pos = features[:, 0:1]  # Position
        pos_normalized = pos / 29903.0  # Normalize by genome length
        pos_log = np.log1p(pos)  # Log-transformed position
        engineered_features.extend([pos_normalized, pos_log])
        
        # Codon and amino acid interactions
        ref_codon = features[:, 1:2]
        alt_codon = features[:, 2:3]
        ref_aa = features[:, 3:4]
        alt_aa = features[:, 4:5]
        
        # Codon change magnitude
        codon_change = np.abs(alt_codon - ref_codon)
        aa_change = np.abs(alt_aa - ref_aa)
        engineered_features.extend([codon_change, aa_change])
        
        # Frequency-based features
        freq = features[:, 8:9]  # Frequency (9th feature)
        freq_log = np.log1p(freq)
        freq_sqrt = np.sqrt(freq)
        engineered_features.extend([freq_log, freq_sqrt])
        
        # Interaction features
        syn_nonsyn = features[:, 5:6]
        transition_transversion = features[:, 6:7]
        cpg_site = features[:, 7:8]
        
        # Mutation type interactions
        mut_type_interaction = syn_nonsyn * transition_transversion
        cpg_freq_interaction = cpg_site * freq
        engineered_features.extend([mut_type_interaction, cpg_freq_interaction])
        
        # Positional conservation score (simplified)
        pos_conservation = np.exp(-pos / 10000.0)  # Higher conservation at functional sites
        engineered_features.append(pos_conservation)
        
        return np.concatenate(engineered_features, axis=1)
    
    def _calculate_feature_stats(self, features: np.ndarray):
        """Calculate and store feature statistics."""
        self.feature_stats = {
            'mean': np.mean(features, axis=0),
            'std': np.std(features, axis=0),
            'min': np.min(features, axis=0),
            'max': np.max(features, axis=0),
            'median': np.median(features, axis=0),
            'q25': np.percentile(features, 25, axis=0),
            'q75': np.percentile(features, 75, axis=0)
        }
    
    def fit(self, features: np.ndarray, labels: Optional[np.ndarray] = None):
        """
        Fit the processor on training data.
        
        Args:
            features: Input features array
            labels: Target labels (optional)
        """
        self.logger.info("Fitting data processor...")
        
        # Handle missing values
        features = self._handle_missing_values(features)
        
        # Engineer features
        features = self._engineer_features(features)
        
        # Calculate feature statistics
        self._calculate_feature_stats(features)
        
        # Fit scaler
        if self.scaler is not None:
            self.scaler.fit(features)
        
        self.fitted = True
        self.logger.info(f"Data processor fitted. Feature shape: {features.shape}")
        
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted processor.
        
        Args:
            features: Input features array
            
        Returns:
            Transformed features
        """
        if not self.fitted:
            raise ValueError("Processor must be fitted before transform")
        
        # Handle missing values
        features = self._handle_missing_values(features)
        
        # Engineer features
        features = self._engineer_features(features)
        
        # Scale features
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        return features
    
    def fit_transform(self, features: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit processor and transform features in one step.
        
        Args:
            features: Input features array
            labels: Target labels (optional)
            
        Returns:
            Transformed features
        """
        self.fit(features, labels)
        return self.transform(features)
    
    def create_sequences(self, features: np.ndarray, labels: np.ndarray, 
                        strain_info: Optional[List] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from features for sequence modeling.
        
        Args:
            features: Input features
            labels: Target labels
            strain_info: Optional strain information for grouping
            
        Returns:
            Sequence features and labels
        """
        if strain_info is None:
            # Simple padding/truncation
            n_samples = len(features)
            if n_samples < self.sequence_length:
                # Pad with zeros
                pad_length = self.sequence_length - n_samples
                features = np.pad(features, ((0, pad_length), (0, 0)), mode='constant')
                labels = np.pad(labels, (0, pad_length), mode='constant')
            else:
                # Truncate
                features = features[:self.sequence_length]
                labels = labels[:self.sequence_length]
            
            return features[None, :, :], labels[None, :]
        else:
            # Group by strain and create sequences
            unique_strains = list(set(strain_info))
            sequences_features = []
            sequences_labels = []
            
            for strain in unique_strains:
                strain_mask = np.array(strain_info) == strain
                strain_features = features[strain_mask]
                strain_labels = labels[strain_mask]
                
                # Pad or truncate
                seq_len = len(strain_features)
                if seq_len < self.sequence_length:
                    pad_length = self.sequence_length - seq_len
                    strain_features = np.pad(strain_features, ((0, pad_length), (0, 0)), mode='constant')
                    strain_labels = np.pad(strain_labels, (0, pad_length), mode='constant')
                else:
                    strain_features = strain_features[:self.sequence_length]
                    strain_labels = strain_labels[:self.sequence_length]
                
                sequences_features.append(strain_features)
                sequences_labels.append(strain_labels)
            
            return np.array(sequences_features), np.array(sequences_labels)
    
    def split_data(self, features: np.ndarray, labels: np.ndarray, 
                   test_size: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and validation sets.
        
        Args:
            features: Input features
            labels: Target labels
            test_size: Fraction for test set (uses validation_split if None)
            
        Returns:
            X_train, X_val, y_train, y_val
        """
        if test_size is None:
            test_size = self.validation_split
        
        return train_test_split(
            features, labels, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=labels if len(np.unique(labels)) > 1 else None
        )
    
    def get_feature_importance_names(self) -> List[str]:
        """Get feature names for interpretation."""
        base_names = [
            'pos', 'ref_codon_idx', 'alt_codon_idx', 'ref_aa_idx', 'alt_aa_idx',
            'syn_nonsyn', 'transition_transversion', 'cpg_site', 'freq'
        ]
        
        if not self.feature_engineering:
            return base_names
        
        engineered_names = base_names + [
            'pos_normalized', 'pos_log', 'codon_change', 'aa_change',
            'freq_log', 'freq_sqrt', 'mut_type_interaction', 
            'cpg_freq_interaction', 'pos_conservation'
        ]
        
        return engineered_names
    
    def get_summary(self) -> Dict[str, Any]:
        """Get processor summary."""
        return {
            'scaling_method': self.scaling_method,
            'sequence_length': self.sequence_length,
            'feature_engineering': self.feature_engineering,
            'handle_missing': self.handle_missing,
            'fitted': self.fitted,
            'feature_stats': self.feature_stats,
            'n_features': len(self.get_feature_importance_names())
        }


class MutationDataset(Dataset):
    """
    PyTorch Dataset for mutation data.
    """
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, 
                 transform: Optional[callable] = None):
        """
        Initialize dataset.
        
        Args:
            features: Input features
            labels: Target labels
            transform: Optional transform function
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features[idx]
        labels = self.labels[idx]
        
        if self.transform:
            features = self.transform(features)
        
        return features, labels


def create_data_loaders(features: np.ndarray, labels: np.ndarray,
                       batch_size: int = 32, validation_split: float = 0.2,
                       random_state: int = 42, num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        features: Input features
        labels: Target labels
        batch_size: Batch size for data loaders
        validation_split: Fraction for validation
        random_state: Random seed
        num_workers: Number of workers for data loading
        
    Returns:
        Training and validation data loaders
    """
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=validation_split, 
        random_state=random_state,
        stratify=labels if len(np.unique(labels)) > 1 else None
    )
    
    # Create datasets
    train_dataset = MutationDataset(X_train, y_train)
    val_dataset = MutationDataset(X_val, y_val)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader
