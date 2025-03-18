"""
Flexible PyTorch matrix factorization models for LensKit.

The components in this package implement several matrix factorization models for
LensKit, and also serve as an example for practical PyTorch recommender
training.

.. stability:: internal
"""

from ._base import FlexMFConfigBase, FlexMFScorerBase
from ._explicit import FlexMFExplicitConfig, FlexMFExplicitScorer

__all__ = [
    "FlexMFConfigBase",
    "FlexMFScorerBase",
    "FlexMFExplicitConfig",
    "FlexMFExplicitScorer",
]
