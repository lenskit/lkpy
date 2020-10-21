"""
TensorFlow-based algorithms.
"""

import logging

from .biasedmf import BiasedMF      # noqa: F401
from .ibmf import IntegratedBiasMF  # noqa: F401
from .bpr import BPR                # noqa: F401

from lenskit.util.parallel import is_mp_worker

try:
    import tensorflow as _tf
    if _tf.__version__ < '2':
        TF_AVAILABLE = False
    else:
        TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

_log = logging.getLogger(__name__)

if TF_AVAILABLE and is_mp_worker():
    _log.info('disabling GPUs in worker process')
    _tf.config.set_visible_devices([], 'GPU')
