from logging import getLogger
from lenskit import util

import tensorflow as tf

_log = getLogger(__name__)


def make_graph(rng_spec=None):
    "Construct a TensorFlow graph (with an optional random seed)"
    rng = util.rng(rng_spec)
    graph = tf.Graph()
    graph.seed = rng.integers(2**31 - 1)
    _log.info('using effective random seed %s (from %s)', graph.seed, rng_spec)
    return graph
