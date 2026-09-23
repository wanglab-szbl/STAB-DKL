"""
Gaussian Process Regression Models for Protein Stability Prediction

This module contains neural network components and GP models
for predicting protein stability changes using Deep Kernel Learning.
"""

import logging
import os
from itertools import chain
from typing import List, Optional, Tuple, Union

import gpytorch
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

    
class Feature_projection(nn.Module):
    """Feature Projection module for feature processing."""
    
    def __init__(self, input_size: int, dropout: float = 0.25):
        """
        
        Args:
            input_size: Input feature dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.fc = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, features]
            
        Returns:
            Output tensor [batch_size, features]
        """
        output = self.dropout(self.fc(x))
        return output


class Mlp(nn.Module):
    """Multi-layer perceptron for dimensionality reduction."""
    
    def __init__(self, input_size: int, out_size: int, dropout: float = 0.25, hidden_size: int = 1280):
        """
        Initialize MLP.
        
        Args:
            input_size: Input feature dimension
            out_size: Output dimension
            dropout: Dropout rate
            hidden_size: Hidden layer size
        """
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, out_size)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP layers."""
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class GPRegressionModel(gpytorch.models.ExactGP):
    """Gaussian Process Regression with neural network feature extractor."""
    
    def __init__(
        self,
        train_x: Optional[torch.Tensor],
        train_y: Optional[torch.Tensor],
        likelihood: gpytorch.likelihoods.Likelihood,
        la: Optional[nn.Module],
        mlp: nn.Module,
        out_dim: int,
        grid_size: int = 100
    ):
        """
        Initialize GP Regression Model.
        
        Args:
            train_x: Training features (can be None for prediction mode)
            train_y: Training labels (can be None for prediction mode)
            likelihood: GP likelihood
            la: Feature_projection network (can be None)
            mlp: MLP for dimensionality reduction
            out_dim: Output dimension
            grid_size: Grid size for interpolation kernel
        """
        super().__init__(train_x, train_y, likelihood)
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=out_dim)),
            num_dims=out_dim,
            grid_size=grid_size
        )
        
        self.la = la
        self.mlp = mlp
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)
    
    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """
        Forward pass through GP model.
        
        Args:
            x: Input tensor
            
        Returns:
            Multivariate normal distribution
        """
        x = self._transform_features(x)
        
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def _transform_features(self, x: torch.Tensor) -> torch.Tensor:
        """Transform input features through LA and MLP."""
        if self.la is not None:
            x = self.la(x)
        
        x = self.mlp(x)
        x = self.scale_to_bounds(x)
        
        return x


class STAB_DKL(nn.Module):
    """
    STAB_DKL: Stability prediction using Deep Kernel Learning.
    
    A deep kernel learning model combining neural network
    with Gaussian Process regression for protein stability prediction.
    Uses an ensemble of GP models for robust uncertainty estimation.
    """
    
    def __init__(
        self,
        input_dim: int,
        out_dim: int = 2,
        grid_size: int = 100,
        dropout: float = 0.5,
        hidden_size: int = 1024,
        device: Optional[torch.device] = None
    ):
        """
        Initialize STAB_DKL model.
        
        Args:
            input_dim: Input feature dimension
            out_dim: MLP output dimension
            grid_size: Grid size for GP interpolation
            dropout: Dropout rate
            hidden_size: Hidden layer size
            device: Computation device
        """
        super().__init__()
        
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.dropout = dropout
        self.hidden_size = hidden_size
        
        # Shared feature extractor
        self.la = Feature_projection(input_dim, dropout=dropout)
        
        # Task-specific MLPs
        self.mlp_ddg = Mlp(input_dim, out_dim, dropout, hidden_size)
        self.mlp_dtm = Mlp(input_dim, out_dim, dropout, hidden_size)
        
        # GP models (created when training data is available)
        self.model_ddg: Optional[GPRegressionModel] = None
        self.model_dtm: Optional[GPRegressionModel] = None
        self.model_ms: Optional[GPRegressionModel] = None
        
        # GP likelihoods
        self.likelihood_ddg = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood_dtm = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood_ms = gpytorch.likelihoods.GaussianLikelihood()
    
    def build_gp_models(
        self,
        train_x_ddg: Optional[torch.Tensor] = None,
        train_y_ddg: Optional[torch.Tensor] = None,
        train_x_dtm: Optional[torch.Tensor] = None,
        train_y_dtm: Optional[torch.Tensor] = None,
        train_x_ms: Optional[torch.Tensor] = None,
        train_y_ms: Optional[torch.Tensor] = None
    ) -> None:
        """
        Build GP models with training data.
        
        Args:
            train_x_ddg: Curated ddG training features
            train_y_ddg: Curated ddG training labels
            train_x_dtm: Curated dTm training features
            train_y_dtm: Curated dTm training labels
            train_x_ms: MegaScale training features
            train_y_ms: MegaScale training labels
        """
        self.model_ddg = GPRegressionModel(
            train_x_ddg, train_y_ddg, self.likelihood_ddg,
            self.la, self.mlp_ddg, self.out_dim, self.grid_size
        ).to(self.device)
        
        self.model_dtm = GPRegressionModel(
            train_x_dtm, train_y_dtm, self.likelihood_dtm,
            self.la, self.mlp_dtm, self.out_dim, self.grid_size
        ).to(self.device)
        
        self.model_ms = GPRegressionModel(
            train_x_ms, train_y_ms, self.likelihood_ms,
            self.la, self.mlp_ddg,  # Share MLP with ddg
            self.out_dim, self.grid_size
        ).to(self.device)
    
    def predict(self, x: Union[torch.Tensor, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions using ensemble of GP models.
        
        Args:
            x: Input features (tensor or numpy array)
            
        Returns:
            Tuple of (mean predictions, variance)
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        
        x = x.to(self.device)
        
        # Set models to eval mode
        self.model_ddg.eval()
        self.model_ms.eval()
        self.likelihood_ddg.eval()
        self.likelihood_ms.eval()
        
        with torch.no_grad():
            # Get predictions from both models
            preds1 = self.likelihood_ddg(self.model_ddg(x))
            preds2 = self.likelihood_ms(self.model_ms(x))
            
            mean1, var1 = preds1.mean, preds1.variance
            mean2, var2 = preds2.mean, preds2.variance
            
            # Weighted average by inverse variance (uncertainty-weighted ensemble)
            w1, w2 = 1 / var1, 1 / var2
            mean = (mean1 * w1 + mean2 * w2) / (w1 + w2)
            var = 1 / (w1 + w2)
        
        return mean, var
    
    def get_trainable_params(self, lr_nn: float = 1e-3, lr_gp: float = 1e-2) -> List[dict]:
        """
        Get parameter groups for optimizer with different learning rates.
        
        Args:
            lr_nn: Learning rate for neural network parameters
            lr_gp: Learning rate for GP parameters
            
        Returns:
            List of parameter groups
        """
        nn_params = chain(
            self.la.parameters(),
            self.mlp_ddg.parameters(),
            self.mlp_dtm.parameters()
        )
        
        gp_params = chain(
            self.model_ddg.covar_module.parameters(),
            self.model_dtm.covar_module.parameters(),
            self.model_ms.covar_module.parameters(),
            self.likelihood_ddg.parameters(),
            self.likelihood_dtm.parameters(),
            self.likelihood_ms.parameters(),
            self.model_ddg.mean_module.parameters(),
            self.model_dtm.mean_module.parameters(),
            self.model_ms.mean_module.parameters()
        )
        
        return [
            {'params': nn_params, 'lr': lr_nn},
            {'params': gp_params, 'lr': lr_gp}
        ]
    
    def save(self, save_file: str) -> None:
        """
        Save model weights to file.
        
        Args:
            save_file: Path to save model
        """
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        
        # Extract GP-specific state dict (excluding LA and MLP)
        def get_gp_state(model):
            full_state = model.state_dict()
            return {k: v for k, v in full_state.items() if not (k.startswith("la.") or k.startswith("mlp."))}
        
        # Save components
        checkpoint = {
            "la": self.la.state_dict(),
            "mlp_ddg": self.mlp_ddg.state_dict(),
            "gp_ddg": get_gp_state(self.model_ddg),
            "gp_ms": get_gp_state(self.model_ms)
        }
        
        torch.save(checkpoint, save_file)
        logger.info(f"Model saved to {save_file}")
    
    def load(self, load_file: str) -> None:
        """
        Load model weights from file.
        
        Args:
            load_file: Path to saved model
        """
        checkpoint = torch.load(load_file, weights_only=False)
        
        # Load shared components
        self.la.load_state_dict(checkpoint["la"])
        self.mlp_ddg.load_state_dict(checkpoint["mlp_ddg"])
        
        # Load GP parameters (strict=False allows missing LA/MLP params)
        self.model_ddg.load_state_dict(checkpoint["gp_ddg"], strict=False)
        self.model_ms.load_state_dict(checkpoint["gp_ms"], strict=False)
        
        logger.info(f"Model loaded from {load_file}")
