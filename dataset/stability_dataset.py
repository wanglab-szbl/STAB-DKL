"""
Dataset and Feature Extraction for Protein Stability Prediction

This module handles data loading, feature extraction, and preprocessing.
"""

import gc
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    feature_paths: Dict[str, str] = field(default_factory=dict)


class TorchStandardScaler:
    """Standard scaler using PyTorch tensors."""
    
    def fit(self, x: torch.Tensor) -> 'TorchStandardScaler':
        """Fit scaler to data."""
        self.mean = x.mean(0, keepdim=True)
        std = x.std(0, keepdim=True)
        self.std = torch.where(std < 1e-8, torch.ones_like(std), std)
        return self
    
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Transform data using fitted scaler."""
        return (x - self.mean) / self.std


class FeatureExtractor:
    """Extract features from protein mutation data."""
    
    def __init__(self, config: DatasetConfig, feature_slice: Dict[str, Tuple[int, int]]):
        """
        Initialize feature extractor.
        
        Args:
            config: Dataset configuration with feature paths
            feature_slice: Dictionary mapping feature names to index ranges
        """
        self.config = config
        self.feature_slice = feature_slice
        
        # Feature data (loaded on demand)
        self.ptmpnn_ft = None
        self.strucmsa_ft = None
        self.esm2_ft = None
        self._loaded = False
        
        # Lookup indices (computed on first use)
        self._ptmpnn_lookup = None
        self._ptmpnn_indices = None
        self._strucmsa_lookup = None
        self._strucmsa_indices = None
        self._esm2_lookup = None
    
    def load_feature_files(self) -> None:
        """Load feature files from disk."""
        if self._loaded:
            return
        
        self.ptmpnn_ft = torch.load(self.config.feature_paths['ptmpnn'], weights_only=False)
        self.strucmsa_ft = torch.load(self.config.feature_paths['strucmsa'], weights_only=False)
        self.esm2_ft = torch.load(self.config.feature_paths['esm2'], weights_only=False)
        
        # Log transform for strucmsa
        self.strucmsa_ft['features'] = torch.log(self.strucmsa_ft['features'] + 0.0005)
        
        if self.ptmpnn_ft is None or self.strucmsa_ft is None or self.esm2_ft is None:
            raise ValueError("Failed to load feature files")
        
        self._loaded = True
    
    def _build_lookups(self) -> None:
        """Build lookup dictionaries for fast feature access."""
        # PTMPNN lookups
        self._ptmpnn_lookup = {
            (seq_id, pos): idx 
            for idx, (seq_id, pos) in enumerate(zip(
                self.ptmpnn_ft['dim0']["seq_id"],
                self.ptmpnn_ft['dim0']["position"]
            ))
        }
        
        # PTMPNN feature indices
        self._ptmpnn_indices = {
            'sqbb_emb': [i for i, x in enumerate(self.ptmpnn_ft['dim1']) if x.startswith(("sqbb_seq", "sqbb_decoder"))],
            'sqbb_prob': {x.split('_')[-1]: i for i, x in enumerate(self.ptmpnn_ft['dim1']) if x.startswith("sqbb_prob") and x.split('_')[-1] != "X"},
            'bb_emb': [i for i, x in enumerate(self.ptmpnn_ft['dim1']) if x.startswith(("bb_seq", "bb_decoder"))],
            'bb_prob': {x.split('_')[-1]: i for i, x in enumerate(self.ptmpnn_ft['dim1']) if x.startswith("bb_prob") and x.split('_')[-1] != "X"}
        }
        
        # StrucMSA lookups
        self._strucmsa_lookup = {
            (seq_id, pos): idx
            for idx, (seq_id, pos) in enumerate(zip(
                self.strucmsa_ft['dim0']["seq_id"],
                self.strucmsa_ft['dim0']["position"]
            ))
        }
        self._strucmsa_indices = {x: i for i, x in enumerate(self.strucmsa_ft['dim1'])}
        
        # ESM2 lookups
        self._esm2_lookup = {
            (seq_id, pos): idx
            for idx, (seq_id, pos) in enumerate(zip(
                self.esm2_ft['dim0']["id"],
                self.esm2_ft['dim0']["position"]
            ))
        }
    
    def extract(self, data_df: pd.DataFrame) -> torch.Tensor:
        """
        Extract features from mutation data.
        
        Args:
            data_df: DataFrame with mutation information
            
        Returns:
            Feature tensor of shape (N, D)
        """
        if not self._loaded:
            self.load_feature_files()
        
        if self._ptmpnn_lookup is None:
            self._build_lookups()
        
        # Validate input
        if data_df.empty:
            raise ValueError("Input DataFrame is empty")
        
        if "seq_id" not in data_df.columns:
            data_df['seq_id'] = data_df['pdb'] + "_" + data_df['chain']
        
        required = ['seq_id', 'position', 'mutAA', 'wtAA']
        missing = [c for c in required if c not in data_df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        data_df = data_df[required]
        
        # Initialize feature tensor
        feature_len = list(self.feature_slice.values())[-1][-1]
        features = torch.empty((len(data_df), feature_len))
        
        # Extract features for each mutation
        for i, row in data_df.iterrows():
            seq_id = row['seq_id']
            position = row['position']
            wtAA = row['wtAA']
            mutAA = row['mutAA']
            
            wt_seq_id = f"{seq_id}_{position}_{wtAA}"
            mut_seq_id = f"{seq_id}_{position}_{mutAA}"
            ptmpnn_idx = self._ptmpnn_lookup[(seq_id, position)]
            strucmsa_idx = self._strucmsa_lookup[(seq_id, position)]
            
            # Extract each feature type
            for feature_id, (start, end) in self.feature_slice.items():
                if feature_id == "ptmpnn_sqbb_difp":
                    prob_idx = self._ptmpnn_indices['sqbb_prob']
                    features[i, start:end] = (
                        self.ptmpnn_ft['features'][ptmpnn_idx, prob_idx[mutAA]] - 
                        self.ptmpnn_ft['features'][ptmpnn_idx, prob_idx[wtAA]]
                    )
                elif feature_id == "ptmpnn_sqbb_emb":
                    features[i, start:end] = self.ptmpnn_ft['features'][ptmpnn_idx, self._ptmpnn_indices['sqbb_emb']]
                elif feature_id == "ptmpnn_sqbb_p":
                    features[i, start:end] = self.ptmpnn_ft['features'][ptmpnn_idx, list(self._ptmpnn_indices['sqbb_prob'].values())]
                elif feature_id == "ptmpnn_bb_difp":
                    prob_idx = self._ptmpnn_indices['bb_prob']
                    features[i, start:end] = (
                        self.ptmpnn_ft['features'][ptmpnn_idx, prob_idx[mutAA]] - 
                        self.ptmpnn_ft['features'][ptmpnn_idx, prob_idx[wtAA]]
                    )
                elif feature_id == "ptmpnn_bb_emb":
                    features[i, start:end] = self.ptmpnn_ft['features'][ptmpnn_idx, self._ptmpnn_indices['bb_emb']]
                elif feature_id == "ptmpnn_bb_p":
                    features[i, start:end] = self.ptmpnn_ft['features'][ptmpnn_idx, list(self._ptmpnn_indices['bb_prob'].values())]
                elif feature_id == "strucmsa_seq":
                    dim1_idx = self._strucmsa_indices
                    features[i, start:end] = (
                        self.strucmsa_ft['features'][strucmsa_idx, dim1_idx[mutAA], 0] - 
                        self.strucmsa_ft['features'][strucmsa_idx, dim1_idx[wtAA], 0]
                    )
                elif feature_id == "strucmsa_struc":
                    dim1_idx = self._strucmsa_indices
                    features[i, start:end] = (
                        self.strucmsa_ft['features'][strucmsa_idx, dim1_idx[mutAA], 1:] - 
                        self.strucmsa_ft['features'][strucmsa_idx, dim1_idx[wtAA], 1:]
                    )
                elif feature_id == "esm2":
                    features[i, start:end] = (
                        self.esm2_ft['features'][self._esm2_lookup[(mut_seq_id, position)], :] - 
                        self.esm2_ft['features'][self._esm2_lookup[(wt_seq_id, position)], :]
                    )
        
        return features


class StabilityDataset:
    """
    Complete dataset manager for protein stability prediction.
    
    Handles training, validation, and test data loading and preprocessing.
    """
    
    def __init__(self, config: DatasetConfig, device: Optional[torch.device] = None, cache_dir: Optional[str] = None):
        """
        Initialize dataset.
        
        Args:
            config: Dataset configuration
            device: Computation device
            cache_dir: Directory for caching processed data
        """
        self.config = config
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = cache_dir
        
        # Preprocessing
        self.scaler = TorchStandardScaler()
        
        # Feature configuration
        self.feature_anno_dt = {
            "ptmpnn_sqbb_difp": ["ptmpnn", "logp", "sqbb", "difp", 1],
            "ptmpnn_sqbb_emb": ["ptmpnn", "sqbb", "emb", 512],
            "ptmpnn_sqbb_p": ["ptmpnn", "logp", "sqbb", 20],
            "ptmpnn_bb_difp": ["ptmpnn", "logp", "bb", "difp", 1],
            "ptmpnn_bb_emb": ["ptmpnn", "bb", "emb", 512],
            "ptmpnn_bb_p": ["ptmpnn", "logp", "bb", 20],
            "strucmsa_seq": ["msa", "msa_seq", 1],
            "strucmsa_struc": ["msa", "msa_struc", 60],
            "esm2": ["esm2", 1280]
        }
        
        # Compute feature slices
        self.feature_slice = {}
        start = 0
        for k, v in self.feature_anno_dt.items():
            dim = v[-1]
            self.feature_slice[k] = (start, start + dim)
            start += dim
        
        # Indices for scaling (exclude embeddings)
        self.scale_idx = []
        for k, (start, end) in self.feature_slice.items():
            if k not in ["ptmpnn_sqbb_emb", "ptmpnn_bb_emb", "esm2"]:
                self.scale_idx.extend(range(start, end))
        
        self.feature_extractor = FeatureExtractor(config, self.feature_slice)
        
        # Data storage
        self.train_data: Dict[str, Dict[str, torch.Tensor]] = {}
        self.valid_data: Dict[str, Dict[str, torch.Tensor]] = {}
        self.test_data: Dict[str, Dict[str, torch.Tensor]] = {}
        self.pred_data: Dict[str, Dict[str, torch.Tensor]] = {}
    
    def load_dataset_config(self, dataset_config: Dict[str, Any]) -> None:
        """
        Load datasets from configuration dictionary.
        
        Args:
            dataset_config: Configuration with train/valid/test/pred dataset paths
        """
        if 'train' in dataset_config:
            self._load_train_datasets(dataset_config['train'])
        
        if 'valid' in dataset_config:
            self._load_eval_datasets(dataset_config['valid'], self.valid_data)
        
        if 'test' in dataset_config:
            self._load_eval_datasets(dataset_config['test'], self.test_data)
        
        if 'pred' in dataset_config:
            self._load_eval_datasets(dataset_config['pred'], self.pred_data)
    
    def _load_train_datasets(self, train_configs: Dict[str, str]) -> None:
        """Load training datasets with label normalization."""
        dataset_types = ["curated_ddg", "curated_dtm", "ms_ddg"]
        ref_std = None
        
        for dataset_type in dataset_types:
            path = train_configs[dataset_type]
            logger.info(f"Loading training dataset ({dataset_type}): {path}")
            df = pd.read_csv(path)
            
            # Normalize labels to reference std
            if dataset_type == "curated_ddg":
                ref_std = df["label"].std()
            df["label"] = df["label"] / df["label"].std() * ref_std
            
            # Extract features
            features = self.feature_extractor.extract(df)
            
            # Fit scaler on first dataset
            if dataset_type == "curated_ddg":
                self.scaler.fit(features[:, self.scale_idx])
            
            # Transform features
            features[:, self.scale_idx] = self.scaler.transform(features[:, self.scale_idx])
            
            # Convert to tensors
            x = features.to(self.device)
            y = torch.tensor(df["label"], dtype=torch.float32).squeeze().to(self.device)
            
            self.train_data[dataset_type] = {'x': x, 'y': y}
            
            # Cleanup
            del df, features
            gc.collect()
    
    def _load_eval_datasets(self, paths: List[str], storage: Dict[str, Dict[str, torch.Tensor]]) -> None:
        """Load evaluation datasets (validation/test/pred)."""
        for path in paths:
            name = Path(path).stem
            logger.info(f"Loading dataset: {name}")
            df = pd.read_csv(path)
            
            # Extract and transform features
            features = self.feature_extractor.extract(df)
            features[:, self.scale_idx] = self.scaler.transform(features[:, self.scale_idx])
            
            # Convert to tensors
            x = features.to(self.device)
            y = torch.tensor(df["label"], dtype=torch.float32).squeeze().to(self.device) if "label" in df.columns else None
            
            storage[name] = {'x': x, 'y': y}
            
            # Cleanup
            del df, features
            gc.collect()
    
    def get_train_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get training tensors for all three tasks.
        
        Returns:
            Tuple of (x1, y1, x2, y2, x3, y3)
        """
        keys = ["curated_ddg", "curated_dtm", "ms_ddg"]
        return tuple(self.train_data[k][v] for k in keys for v in ['x', 'y'])
    
    def get_input_dim(self) -> int:
        """Get input feature dimension."""
        for data in [self.train_data, self.valid_data, self.test_data]:
            if data:
                first_key = list(data.keys())[0]
                return data[first_key]['x'].shape[-1]
    
    def save(self, save_path: str) -> None:
        """Save scaler and training data for prediction."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Exclude curated_dtm from saved data
        train_data_new = {k: v for k, v in self.train_data.items() if k != "curated_dtm"}
        
        torch.save({
            "scaler": {"mean": self.scaler.mean, "std": self.scaler.std},
            "data": train_data_new
        }, save_path)
        
        logger.info(f"Dataset metadata saved to {save_path}")
    
    def load(self, load_path: str) -> None:
        """Load scaler and training data for prediction."""
        data = torch.load(load_path, weights_only=False)
        self.scaler.mean = data["scaler"]["mean"]
        self.scaler.std = data["scaler"]["std"]
        self.train_data = data['data']
        
        logger.info(f"Dataset metadata loaded from {load_path}")
