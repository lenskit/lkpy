"""
LensKit statistical computations.
"""

import numpy as np
from numpy.typing import ArrayLike


def damped_mean(xs: ArrayLike, damping: float | None) -> float:
    xs = np.asarray(xs)
    if damping is not None and damping > 0:
        return xs.sum().item() / (np.sum(np.isfinite(xs)).item() + damping)
    else:
        return xs.mean().item()
