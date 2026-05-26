"""
Protein Stability Prediction Pipeline

This script combines feature extraction and prediction into a single pipeline.

Usage:
    python predict_pipeline.py \
        --input mutations.csv \
        --output predictions.csv \
        --pdb_dir data/pdbs/ \
        --model_path weights/model.pt \
        --data_path weights/data.pt \
        --trainer_config weights/trainer_config.pt \
        --esm2_path /path/to/esm2 \
        --mmseqs_db /path/to/db \
        --device cuda:0
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile
import yaml
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import torch

from get_features import (
    get_strucmsa_features,
    get_esm2_features,
    get_ptmpnn_features
)

from model import STAB_DKL
from dataset import StabilityDataset, DatasetConfig
from train import Trainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PredictionPipeline:
    """Pipeline for protein stability prediction from mutation data."""
    
    def __init__(self, args: argparse.Namespace):
        """
        Initialize prediction pipeline.
        
        Args:
            args: Command line arguments
        """
        self.args = args
        self.tmp_dir = None
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
    def run(self) -> pd.DataFrame:
        """
        Run the complete prediction pipeline.
        
        Returns:
            DataFrame with predictions
        """
        try:
            # Create temporary directory
            self.tmp_dir = tempfile.mkdtemp(prefix="stab_pred_")
            logger.info(f"Created temporary directory: {self.tmp_dir}")
            
            # Step 1: Extract features
            logger.info("\n" + "="*60)
            logger.info("STEP 1: Feature Extraction")
            logger.info("="*60)
            self._extract_features()
            
            # Step 2: Predict
            logger.info("\n" + "="*60)
            logger.info("STEP 2: Prediction")
            logger.info("="*60)
            result_df = self._predict()
            
            return result_df
            
        finally:
            # Cleanup temporary directory
            if self.tmp_dir and os.path.exists(self.tmp_dir):
                logger.info(f"\nCleaning up temporary directory: {self.tmp_dir}")
                shutil.rmtree(self.tmp_dir)
    
    def _extract_features(self) -> None:
        """Extract features (PTMPNN, StrucMSA, ESM2)."""
        # Load input data
        logger.info(f"Loading input data from: {self.args.input}")
        df = pd.read_csv(self.args.input)
        
        # Process seq_id and chain
        if "seq_id" not in df.columns:
            df['seq_id'] = df.apply(lambda x: x['pdb'] + "_" + x['chain'], axis=1)
        
        if self.args.force_chain_A:
            df['chain'] = 'A'
        
        # Add pdb_file path (only if pdb_file column not in input)
        if "pdb_file" not in df.columns:
            if self.args.pdb_dir is None:
                raise ValueError("pdb_file column not found in input and pdb_dir is not provided")
            df['pdb_file'] = df.apply(
                lambda x: os.path.join(self.args.pdb_dir, x['seq_id'].lower() + "_model.pdb"),
                axis=1
            )
        else:
            logger.info("Using pdb_file column from input data")
        
        logger.info(f"Total mutations to process: {len(df)}")
        
        # Setup MSA file
        if self.args.msa_file:
            # Use provided MSA file
            msa_file = self.args.msa_file
            logger.info(f"Using provided MSA file: {msa_file}")
        else:
            # Generate MSA in temporary directory
            msa_dir = os.path.join(self.tmp_dir, "msa")
            os.makedirs(msa_dir, exist_ok=True)
            msa_file = os.path.join(msa_dir, "msa.fa")
            logger.info(f"Will generate MSA at: {msa_file}")
        
        # MSA generation config
        mmseqs_config = {
            'mmseqs_cmd': self.args.mmseqs_cmd,
            'mmseqs_db': self.args.mmseqs_db,
            'use_gpu': self.args.use_gpu,
            'max_seqs': self.args.max_seqs,
            'threads': self.args.threads
        }
        
        # Validate MSA settings
        if not os.path.exists(msa_file) and self.args.mmseqs_db is None:
            raise ValueError("MSA file does not exist and mmseqs_db is not provided")
        
        # Feature extraction config
        feat_config = {'n_jobs': self.args.n_jobs}
        
        # Output prefix (files will be: ptmpnn.pt, strucmsa.pt, esm2.pt)
        output_prefix = os.path.join(self.tmp_dir, "features_")
        
        # Extract StrucMSA features
        logger.info("\n[1/3] Extracting StrucMSA features...")
        get_strucmsa_features(df, msa_file, output_prefix, mmseqs_config, feat_config['n_jobs'])
        
        # Extract ESM2 features
        logger.info("\n[2/3] Extracting ESM2 features...")
        get_esm2_features(df, self.args.esm2_path, output_prefix)
        torch.cuda.empty_cache()
        
        # Extract PTMPNN features
        logger.info("\n[3/3] Extracting PTMPNN features...")
        get_ptmpnn_features(df, output_prefix)
        
        logger.info("\nFeature extraction completed!")
        logger.info(f"Features saved to: {self.tmp_dir}")
    
    def _predict(self) -> pd.DataFrame:
        """Run prediction using extracted features."""
        # Initialize dataset
        feature_prefix = os.path.join(self.tmp_dir, "features_")
        
        dataset_cfg = DatasetConfig(
            feature_paths={
                "ptmpnn": f"{feature_prefix}ptmpnn.pt",
                "strucmsa": f"{feature_prefix}strucmsa.pt",
                "esm2": f"{feature_prefix}esm2.pt"
            }
        )
        
        dataset = StabilityDataset(config=dataset_cfg, device=self.device)
        
        # Load saved scaler and training data
        dataset.load(self.args.data_path)
        logger.info(f"Loaded dataset scaler from: {self.args.data_path}")
        
        # Load prediction data
        dataset.load_dataset_config({"pred": [self.args.input]})
        logger.info(f"Loaded prediction data from: {self.args.input}")
        
        # Initialize model
        model = STAB_DKL(input_dim=dataset.get_input_dim(), device=self.device)
        
        # Build GP models with dummy training data
        train_data = dataset.train_data
        model.build_gp_models(
            train_x_ddg=train_data["curated_ddg"]['x'],
            train_y_ddg=train_data["curated_ddg"]['y'],
            train_x_dtm=train_data['ms_ddg']['x'],
            train_y_dtm=train_data['ms_ddg']['y'],
            train_x_ms=train_data['ms_ddg']['x'],
            train_y_ms=train_data['ms_ddg']['y']
        )
        
        # Load trained weights
        model.load(self.args.model_path)
        logger.info(f"Loaded model from: {self.args.model_path}")
        
        # Initialize trainer and load config
        trainer = Trainer(model, dataset, None, self.device)
        trainer.load_config(self.args.trainer_config)
        logger.info(f"Loaded trainer config from: {self.args.trainer_config}")
        
        # Generate predictions
        new_pred_data = {}
        data = list(dataset.pred_data.values())[0]['x']
        data_chunk = torch.split(data, self.args.batch_size, dim = 0)
        new_pred_data = {f'chunk_{i}': {"x": chunk, "y": None} for i, chunk in enumerate(data_chunk)}
        trainer.dataset.pred_data = new_pred_data
        result_df = trainer.pred()  
        
        # Load original data and add predictions
        data_df = pd.read_csv(self.args.input)
        data_df['pred'] = result_df['pred']
        data_df['var'] = result_df['var']
        
        # Add seq_id if not present
        if "seq_id" not in data_df.columns:
            data_df['seq_id'] = data_df['pdb'] + "_" + data_df['chain']
        
        # Select output columns
        output_cols = ['seq_id', 'chain', 'position', 'wtAA', 'mutAA', 'pred', 'var']
        data_df = data_df[output_cols]
        
        # Save predictions
        os.makedirs(os.path.dirname(self.args.output) if os.path.dirname(self.args.output) else '.', exist_ok=True)
        data_df.to_csv(self.args.output, index=False)
        
        logger.info(f"\nPredictions saved to: {self.args.output}")
        logger.info(f"Output columns: {data_df.columns.tolist()}")
        
        return data_df


def parse_args() -> argparse.Namespace:
    """Parse command line arguments or load from config file."""
    parser = argparse.ArgumentParser(
        description="Protein stability prediction pipeline: feature extraction + prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input/Output
    parser.add_argument('--input', help='Input CSV file with mutations')
    parser.add_argument('--output', help='Output CSV file for predictions')
    parser.add_argument('--pdb_dir', default=None, help='Directory containing PDB files (optional if pdb_file column exists)')
    parser.add_argument('--msa_file', default=None, help='Path to existing MSA file (optional, will generate if not provided)')
    
    # Model files
    parser.add_argument('--model_path', help='Trained model path')
    parser.add_argument('--data_path', help='Dataset scaler path')
    parser.add_argument('--trainer_config', help='Trainer config path')
    
    # ESM2
    parser.add_argument('--esm2_path', help='Path to ESM2 model')
    
    # MSA generation (optional if msa_file provided)
    parser.add_argument('--mmseqs_cmd', default='mmseqs', help='Path to mmseqs2 executable')
    parser.add_argument('--mmseqs_db', default=None, help='Path to mmseqs2 database (required if msa_file not provided)')
    parser.add_argument('--use_gpu', action='store_true', default=True, help='Use GPU for MSA generation')
    parser.add_argument('--max_seqs', type=int, default=1000, help='Max sequences for MSA')
    parser.add_argument('--threads', type=int, default=22, help='Number of threads')
    
    # Feature extraction
    parser.add_argument('--n_jobs', type=int, default=22, help='Number of parallel jobs for feature extraction')
    
    # Chain handling
    parser.add_argument('--force_chain_A', action='store_true', default=False, help='Force chain to A')
    
    # Device
    parser.add_argument('--device', default='cuda:0', help='Device to use')

    # batch size 
    parser.add_argument('--batch_size', type = int, default=1024, help='Batch size')
    
    args = parser.parse_args()
    
    # Validate required arguments
    required_args = ['input', 'output', 'model_path', 'data_path', 'trainer_config', 'esm2_path']
    missing = [arg for arg in required_args if getattr(args, arg) is None]
    if missing:
        parser.error(f"The following arguments are required: {', '.join('--' + arg for arg in missing)}")
    
    return args


def main():
    """Main entry point."""
    args = parse_args()
    
    logger.info("="*60)
    logger.info("Protein Stability Prediction Pipeline")
    logger.info("="*60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    if args.pdb_dir:
        logger.info(f"PDB directory: {args.pdb_dir}")
    else:
        logger.info("Using pdb_file column from input data")
    if args.msa_file:
        logger.info(f"MSA file: {args.msa_file}")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch_szie: {args.batch_size}")
    logger.info("="*60)
    
    # Run pipeline
    pipeline = PredictionPipeline(args)
    result_df = pipeline.run()
    
    logger.info("\n" + "="*60)
    logger.info("Pipeline completed successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
