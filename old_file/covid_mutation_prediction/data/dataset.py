"""
PyTorch datasets for COVID-19 mutation prediction.

This module provides dataset classes for handling mutation data
in PyTorch training pipelines.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Callable, List, Dict, Any
from sklearn.preprocessing import StandardScaler
import logging


class AdvancedMutationDataset(Dataset):
    """
    Advanced PyTorch dataset for mutation prediction with data augmentation
    and preprocessing capabilities.
    """
    
    def __init__(self, 
                 features: np.ndarray,
                 labels: np.ndarray,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 augmentation: bool = False,
                 feature_names: Optional[List[str]] = None):
        """
        Initialize the mutation dataset.
        
        Args:
            features: Input features array
            labels: Target labels array
            transform: Optional transform to apply to features
            target_transform: Optional transform to apply to labels
            augmentation: Whether to apply data augmentation
            feature_names: Names of features for interpretation
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.transform = transform
        self.target_transform = target_transform
        self.augmentation = augmentation
        self.feature_names = feature_names or [f'feature_{i}' for i in range(features.shape[-1])]
        
        self.logger = logging.getLogger(__name__)
        
        # Validate data
        if len(self.features) != len(self.labels):
            raise ValueError("Features and labels must have the same length")
    
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (features, label)
        """
        features = self.features[idx]
        label = self.labels[idx]
        
        # Apply augmentation during training
        if self.augmentation:
            features = self._apply_augmentation(features)
        
        # Apply transforms
        if self.transform:
            features = self.transform(features)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return features, label
    
    def _apply_augmentation(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply data augmentation to features.
        
        Args:
            features: Input features
            
        Returns:
            Augmented features
        """
        # Add small random noise
        if torch.rand(1) < 0.3:  # 30% chance
            noise = torch.randn_like(features) * 0.01
            features = features + noise
        
        # Feature dropout (randomly set some features to zero)
        if torch.rand(1) < 0.2:  # 20% chance
            dropout_mask = torch.rand_like(features) > 0.1
            features = features * dropout_mask
        
        return features
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset."""
        if len(self.labels.shape) == 1:
            # Binary classification
            unique, counts = torch.unique(self.labels, return_counts=True)
            return {f'class_{int(u)}': int(c) for u, c in zip(unique, counts)}
        else:
            # Multi-label classification
            return {f'label_{i}': int(torch.sum(self.labels[:, i])) 
                   for i in range(self.labels.shape[1])}
    
    def get_feature_statistics(self) -> pd.DataFrame:
        """Get statistics for each feature."""
        stats = []
        for i, name in enumerate(self.feature_names):
            if len(self.features.shape) == 2:
                feature_data = self.features[:, i]
            else:
                feature_data = self.features[:, :, i].flatten()
            
            stats.append({
                'feature_name': name,
                'mean': float(torch.mean(feature_data)),
                'std': float(torch.std(feature_data)),
                'min': float(torch.min(feature_data)),
                'max': float(torch.max(feature_data)),
                'median': float(torch.median(feature_data))
            })
        
        return pd.DataFrame(stats)


class SequenceMutationDataset(Dataset):
    """
    Dataset for sequence-based mutation prediction with variable length sequences.
    """
    
    def __init__(self,
                 sequences: List[List],
                 labels: np.ndarray,
                 max_length: int = 1000,
                 padding_value: float = 0.0,
                 feature_vocabs: Optional[List[Dict]] = None):
        """
        Initialize sequence dataset.
        
        Args:
            sequences: List of sequences (each sequence is a list of mutations)
            labels: Target labels
            max_length: Maximum sequence length for padding
            padding_value: Value to use for padding
            feature_vocabs: Vocabularies for encoding features
        """
        self.sequences = sequences
        self.labels = torch.FloatTensor(labels)
        self.max_length = max_length
        self.padding_value = padding_value
        self.feature_vocabs = feature_vocabs
        
        # Encode sequences if vocabs are provided
        if feature_vocabs:
            self.encoded_sequences = self._encode_sequences()
        else:
            self.encoded_sequences = self._pad_sequences()
    
    def _encode_sequences(self) -> torch.Tensor:
        """Encode sequences using feature vocabularies."""
        encoded = []
        
        for sequence in self.sequences:
            # Encode each mutation in the sequence
            encoded_sequence = []
            for mutation in sequence:
                encoded_mutation = []
                for i, (feature_val, vocab) in enumerate(zip(mutation, self.feature_vocabs)):
                    encoded_val = vocab.get(str(feature_val), vocab.get('<UNK>', 0))
                    encoded_mutation.append(encoded_val)
                encoded_sequence.append(encoded_mutation)
            
            # Pad sequence to max_length
            while len(encoded_sequence) < self.max_length:
                encoded_sequence.append([self.padding_value] * len(self.feature_vocabs))
            
            # Truncate if too long
            encoded_sequence = encoded_sequence[:self.max_length]
            encoded.append(encoded_sequence)
        
        return torch.LongTensor(encoded)
    
    def _pad_sequences(self) -> torch.Tensor:
        """Pad sequences without encoding."""
        padded = []
        
        for sequence in self.sequences:
            # Convert to tensor
            if isinstance(sequence[0], (list, tuple)):
                seq_tensor = torch.FloatTensor(sequence)
            else:
                seq_tensor = torch.FloatTensor([[x] for x in sequence])
            
            # Pad or truncate
            seq_len, feature_dim = seq_tensor.shape
            
            if seq_len < self.max_length:
                # Pad
                padding = torch.full((self.max_length - seq_len, feature_dim), 
                                   self.padding_value)
                seq_tensor = torch.cat([seq_tensor, padding], dim=0)
            else:
                # Truncate
                seq_tensor = seq_tensor[:self.max_length]
            
            padded.append(seq_tensor)
        
        return torch.stack(padded)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoded_sequences[idx], self.labels[idx]


class TimeSeriesMutationDataset(Dataset):
    """
    Dataset for time-series mutation prediction with temporal features.
    """
    
    def __init__(self,
                 time_series_data: Dict[int, List],
                 labels: Dict[int, np.ndarray],
                 sequence_length: int = 50,
                 prediction_horizon: int = 1):
        """
        Initialize time series dataset.
        
        Args:
            time_series_data: Dictionary mapping time steps to mutation data
            labels: Dictionary mapping time steps to labels
            sequence_length: Length of input sequences
            prediction_horizon: Number of steps to predict ahead
        """
        self.time_series_data = time_series_data
        self.labels = labels
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Create samples
        self.samples = self._create_samples()
    
    def _create_samples(self) -> List[Tuple[List, np.ndarray]]:
        """Create samples from time series data."""
        samples = []
        time_steps = sorted(self.time_series_data.keys())
        
        for i in range(len(time_steps) - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence
            input_sequence = []
            for j in range(self.sequence_length):
                time_step = time_steps[i + j]
                input_sequence.extend(self.time_series_data[time_step])
            
            # Target
            target_time_step = time_steps[i + self.sequence_length + self.prediction_horizon - 1]
            target = self.labels[target_time_step]
            
            samples.append((input_sequence, target))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence, label = self.samples[idx]
        return torch.FloatTensor(sequence), torch.FloatTensor(label)


def create_balanced_loader(dataset: Dataset, batch_size: int = 32,
                          num_workers: int = 0, pin_memory: bool = True) -> DataLoader:
    """
    Create a balanced data loader for imbalanced datasets.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        
    Returns:
        Balanced DataLoader
    """
    from torch.utils.data import WeightedRandomSampler
    
    # Calculate class weights for balanced sampling
    labels = dataset.labels
    if len(labels.shape) == 1:
        # Binary classification
        class_counts = torch.bincount(labels.long())
        weights = 1.0 / class_counts.float()
        sample_weights = weights[labels.long()]
    else:
        # Multi-label - use sum of positive labels as weight
        label_sums = torch.sum(labels, dim=1)
        sample_weights = 1.0 / (label_sums + 1e-8)  # Add small epsilon
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


def collate_variable_length(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for variable length sequences.
    
    Args:
        batch: List of (sequence, label) tuples
        
    Returns:
        Padded sequences, labels, and lengths
    """
    sequences, labels = zip(*batch)
    
    # Get lengths
    lengths = torch.LongTensor([len(seq) for seq in sequences])
    
    # Pad sequences
    max_len = max(lengths)
    padded_sequences = []
    
    for seq in sequences:
        if len(seq) < max_len:
            padding = torch.zeros(max_len - len(seq), seq.shape[-1])
            padded_seq = torch.cat([seq, padding], dim=0)
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)
    
    return (torch.stack(padded_sequences), 
            torch.stack(labels), 
            lengths)
