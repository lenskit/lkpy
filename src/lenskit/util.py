import warnings

from lenskit.logging import Stopwatch
from lenskit.random import derivable_rng, random_generator, set_global_rng

warnings.warn("lenskit.util is deprecated, import from original modules", DeprecationWarning)

__all__ = [
    "Stopwatch",
    "derivable_rng",
    "random_generator",
    "set_global_rng",
]
