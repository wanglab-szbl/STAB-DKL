"""
ESM2 Feature Extraction Module
"""

from .esm2_features import (
    get_embeddings,
    load_model,
    clear_cache
)

__all__ = [
    "get_embeddings",
    "load_model",
    "clear_cache"
]
