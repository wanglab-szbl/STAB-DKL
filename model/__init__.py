"""
Model module for protein stability prediction.

STAB_DKL: Stability prediction using Deep Kernel Learning
"""

from .gp_model import (
    Feature_projection,
    Mlp,
    GPRegressionModel,
    STAB_DKL
)

__all__ = [
    'Feature_projection',
    'Mlp',
    'GPRegressionModel',
    'STAB_DKL'
]
