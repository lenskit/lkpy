"""
TensorFlow-based algorithms.
"""

import logging

from .util import have_usable_tensorflow
from .biasedmf import BiasedMF      # noqa: F401
from .ibmf import IntegratedBiasMF  # noqa: F401
from .bpr import BPR                # noqa: F401

from lenskit.util.parallel import is_mp_worker

TF_AVAILABLE = have_usable_tensorflow()

_log = logging.getLogger(__name__)

if TF_AVAILABLE and is_mp_worker():
    import tensorflow as _tf
    _log.info('disabling GPUs in worker process')
    _tf.config.set_visible_devices([], 'GPU')
    _log.info('disabling multithreading in worker')
    _tf.config.threading.set_inter_op_parallelism_threads(1)
    _tf.config.threading.set_intra_op_parallelism_threads(1)
