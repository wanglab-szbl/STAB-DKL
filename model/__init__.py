"""
Model module for protein stability prediction.

STAB_DKL: Stability prediction using Deep Kernel Learning
"""

from .gp_model import (
    LightAttention,
    Mlp,
    GPRegressionModel,
    STAB_DKL
)

__all__ = [
    'LightAttention',
    'Mlp',
    'GPRegressionModel',
    'STAB_DKL'
]
