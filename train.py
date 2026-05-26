"""
Protein Stability Prediction - Training Script

Usage:
    python train.py  # Train with default config
    python train.py --config config/train.yaml  # Train with custom config
"""

import argparse
import gc
import logging
import os
from copy import deepcopy
from typing import Dict, Optional, Tuple

import gpytorch
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torchmetrics.functional import pearson_corrcoef, spearman_corrcoef
from tqdm import tqdm

from model import STAB_DKL
from dataset import StabilityDataset, DatasetConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """Training manager for STAB_DKL model."""
    
    def __init__(self, model: STAB_DKL, dataset: StabilityDataset, config: dict, device: str = "cuda:0"):
        """
        Initialize trainer.
        
        Args:
            model: STAB_DKL model instance
            dataset: StabilityDataset instance
            config: Training configuration dictionary
            device: Computation device
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = device
        
        # Training components (initialized in setup)
        self.optimizer = None
        self.scheduler = None
        self.mlls = []
        self.scale_pred = 1.0
    
    def setup(self) -> None:
        """Setup optimizer, scheduler, and marginal log likelihoods."""
        # Get training data
        x1, y1, x2, y2, x3, y3 = self.dataset.get_train_tensors()
        
        # Build GP models
        self.model.build_gp_models(x1, y1, x2, y2, x3, y3)
        
        # Setup optimizer
        train_cfg = self.config['training']
        param_groups = self.model.get_trainable_params(
            lr_nn=float(train_cfg['lr']),
            lr_gp=float(train_cfg.get('lr_gp', 5e-2))
        )
        
        self.optimizer = torch.optim.Adam(
            param_groups,
            betas=tuple(train_cfg.get('betas', [0.9, 0.99]))
        )
        
        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=train_cfg['epochs'],
            eta_min=1e-6
        )
        
        # Setup marginal log likelihoods for each task
        self.mlls = [
            gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood_ddg, self.model.model_ddg),
            gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood_dtm, self.model.model_dtm),
            gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood_ms, self.model.model_ms)
        ]
    
    def train(self, save_path: Optional[str] = None, best_score: float = 0.0, save_file: Optional[str] = None) -> Tuple[float, str]:
        """
        Train the model with multiple learning rates.
        
        Args:
            save_path: Base path for saving models
            best_score: Best validation score so far
            save_file: Path to current best model file
            
        Returns:
            Tuple of (best_score, save_file)
        """
        train_cfg = self.config['training']
        epochs = train_cfg['epochs']
        min_save_epoch = train_cfg.get('min_save_epoch', 30)
        
        # Get training data
        x1, y1, x2, y2, x3, y3 = self.dataset.get_train_tensors()
        
        pbar = tqdm(range(epochs), desc="Training")
        best_epoch = 0
        
        for epoch in pbar:
            # Training step
            loss = self._train_step(x1, y1, x2, y2, x3, y3)
            
            # Validation
            score = self._validate()
            
            # Save best model
            if score > best_score and (epoch + 1) >= min_save_epoch:
                best_score = score
                best_epoch = epoch + 1
                
                # Remove old model file
                if save_file and os.path.exists(save_file):
                    os.remove(save_file)
                
                # Save new model with score in filename
                save_file = f"{save_path}_{round(best_score.item(), 3)}.pt"
                self.model.save(save_file)
            
            pbar.set_postfix({
                'loss': f"{loss:.4f}",
                'best': f"{score:.4f}/{best_score:.4f}@{best_epoch}"
            })
        
        return best_score, save_file
    
    def _train_step(
        self,
        x1: torch.Tensor, y1: torch.Tensor,
        x2: torch.Tensor, y2: torch.Tensor,
        x3: torch.Tensor, y3: torch.Tensor
    ) -> float:
        """Execute single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Compute weighted losses for three tasks
        out1 = self.model.model_ddg(x1)
        loss = -self.mlls[0](out1, y1) * 0.25
        
        out2 = self.model.model_dtm(x2)
        loss += -self.mlls[1](out2, y2) * 0.5
        
        out3 = self.model.model_ms(x3)
        loss += -self.mlls[2](out3, y3) * 0.25
        
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def _validate(self) -> float:
        """Validate model and return score."""
        self.model.eval()
        metrics = []
        
        with torch.no_grad():
            for name, data in self.dataset.valid_data.items():
                x, y = data['x'], data['y']
                pred, var = self.model.predict(x)
                
                pearson = pearson_corrcoef(y, pred)
                spearman = spearman_corrcoef(y, pred)
                
                metrics.append({
                    'name': name,
                    'pearson': pearson,
                    'spearman': spearman
                })
        
        metric_df = pd.DataFrame(metrics)
        score = metric_df['pearson'].sum() + metric_df['spearman'].sum()
        
        return score
    
    def find_scale_factor(self) -> torch.Tensor:
        """Find optimal scale factor for predictions via grid search."""
        self.model.eval()
        
        preds = []
        labels = []
        
        with torch.no_grad():
            for data in self.dataset.valid_data.values():
                pred, _ = self.model.predict(data['x'])
                preds.append(pred)
                labels.append(data['y'])
        
        all_pred = torch.cat(preds)
        all_label = torch.cat(labels)
        
        # Grid search for optimal scale
        min_mse = float('inf')
        best_scale = torch.tensor(1.0)
        
        for s in [1.0 + i * 0.1 for i in range(11)]:
            s_tensor = torch.tensor(s)
            mse = F.mse_loss(all_pred / s_tensor, all_label).item()
            
            if mse < min_mse * 0.9:  # Significant improvement threshold
                min_mse = mse
                best_scale = s_tensor
        
        self.scale_pred = best_scale
        logger.info(f"Optimal scale factor: {best_scale:.2f}")
        
        return best_scale
    
    def test(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Evaluate model on test set.
        
        Returns:
            Tuple of (metrics_df, predictions_df)
        """
        self.model.eval()
        metrics = []
        predictions = []
        
        with torch.no_grad():
            for name, data in self.dataset.test_data.items():
                x, y = data['x'], data['y']
                pred, var = self.model.predict(x)
                pred = pred / self.scale_pred
                
                # Calculate metrics
                pearson = pearson_corrcoef(y, pred)
                spearman = spearman_corrcoef(y, pred)
                mse = F.mse_loss(pred, y).item()
                mae = F.l1_loss(pred, y).item()
                
                res = torch.abs(pred - y)
                pear_res = pearson_corrcoef(var, res)
                spear_res = spearman_corrcoef(var, res)
                
                metrics.append({
                    'dataset': name,
                    'rmse': round(mse ** 0.5, 4),
                    'mae': round(mae, 4),
                    'pear': round(pearson.item(), 4),
                    'spear': round(spearman.item(), 4),
                    'pear_res': round(pear_res.item(), 4),
                    'spear_res': round(spear_res.item(), 4)
                })
                
                predictions.append(pd.DataFrame({
                    'dataset': name,
                    'pred': pred.cpu().numpy(),
                    'var': var.cpu().numpy(),
                    'label': y.cpu().numpy()
                }))
        
        return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)
    
    def pred(self) -> pd.DataFrame:
        """
        Generate predictions for prediction dataset.
        
        Returns:
            DataFrame with predictions
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for name, data in self.dataset.pred_data.items():
                pred, var = self.model.predict(data['x'])
                pred = pred / self.scale_pred
                
                predictions.append(pd.DataFrame({
                    'dataset': name,
                    'pred': pred.cpu().numpy(),
                    'var': var.cpu().numpy(),
                }))
        
        if len(predictions) == 1:
            return predictions[0]
        return pd.concat(predictions, ignore_index=True)
    
    def save_config(self, save_file: str) -> None:
        """Save scale factor and config."""
        torch.save({
            "scale": self.scale_pred,
            "config": self.config
        }, save_file)
    
    def load_config(self, load_file: str) -> None:
        """Load scale factor and config."""
        data = torch.load(load_file, weights_only=False)
        self.scale_pred = data['scale']
        self.config = data['config']


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_experiment(config: dict, seed: int, device: torch.device) -> None:
    """
    Run single training experiment with multiple learning rates.
    
    Args:
        config: Configuration dictionary
        seed: Random seed
        device: Computation device
    """
    from pytorch_lightning import seed_everything
    
    # Set random seed
    seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    exp_cfg = config['experiment']
    save_dir = exp_cfg['save_dir']
    
    # Initialize dataset
    dataset_cfg = DatasetConfig(feature_paths=config['data']['feature_paths'])
    dataset = StabilityDataset(dataset_cfg, device=device, cache_dir=exp_cfg.get('cache_dir'))
    
    # Load data
    dataset.load_dataset_config(config['datasets'])
    dataset.save(f"{save_dir}/data.pt")
    
    # Train with multiple learning rates
    lr_list = config['training']['lr']
    best_score = 0.0
    save_file = None
    config_new = deepcopy(config)
    
    for lr in lr_list:
        logger.info(f"\n{'='*60}\nTraining with seed {seed}, lr {lr}\n{'='*60}")
        config_new['training']['lr'] = lr
        
        # Initialize model
        model_cfg = config_new['model']
        model = STAB_DKL(
            input_dim=dataset.get_input_dim(),
            out_dim=model_cfg['out_dim'],
            grid_size=model_cfg['grid_size'],
            dropout=model_cfg['dropout'],
            hidden_size=model_cfg['hidden_size'],
            device=device
        )
        
        # Train
        trainer = Trainer(model, dataset, config_new, device)
        trainer.setup()
        save_path = f"{save_dir}/model_{seed}"
        best_score, save_file = trainer.train(save_path=save_path, best_score=best_score, save_file=save_file)
    
    # Load best model and compute scale factor
    model.load(save_file)
    trainer.find_scale_factor()
    trainer.save_config(f"{save_dir}/trainer_config_{seed}.pt")
    
    # Test and save results
    metrics_df, pred_df = trainer.test()
    
    result_dir = exp_cfg['result_dir']
    os.makedirs(result_dir, exist_ok=True)
    
    metrics_df.to_csv(f"{result_dir}/metrics_seed_{seed}.csv", index=False)
    pred_df.to_csv(f"{result_dir}/predictions_seed_{seed}.csv", index=False)
    
    logger.info(f"\nTest Results (seed {seed}):\n{metrics_df}")
    
    # Cleanup
    del model, dataset, trainer
    gc.collect()
    torch.cuda.empty_cache()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train protein stability model')
    parser.add_argument('--config', default='config/train.yaml', help='Path to configuration file')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create save directory
    save_dir = config['experiment']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(config.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"Using device: {device}")
    
    # Run experiments for each seed
    seeds = config['experiment']['seeds']
    for seed in seeds:
        run_experiment(config, seed, device)
    
    logger.info("\nAll experiments completed!")


if __name__ == "__main__":
    main()
