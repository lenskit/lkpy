"""
FunkSVD (biased MF).
"""

from collections import namedtuple
import logging

import pandas as pd

from .. import util as lku
from .. import check
from baselines import Bias

_logger = logging.getLogger(__package__)


class FunkSVD:
    def __init__(self, features, lrate=0.001, reg=0.02, damping=5):
        self.features = features
        self.learning_rate = lrate
        self.regularization = reg
        self.damping = damping

    def train(self, ratings, bias=None):
        ""
        if bias is None:
            _logger.info('training bias model')
            bias = Bias(damping=self.damping).train(ratings)
