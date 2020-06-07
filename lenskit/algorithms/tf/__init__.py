"""
TensorFlow-based algorithms.
"""

import logging
import tensorflow as _tf

from .biasedmf import BiasedMF
from .ibmf import IntegratedBiasMF
from .bpr import BPR

from lenskit.util.parallel import is_worker

_log = logging.getLogger(__name__)

if is_worker():
    _log.info('disabling GPUs in worker process')
    _tf.config.set_visible_devices([], 'GPU')
