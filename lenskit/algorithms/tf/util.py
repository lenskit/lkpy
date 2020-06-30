from logging import getLogger
from lenskit import util

try:
    import tensorflow as tf
except ImportError:
    tf = None

_log = getLogger(__name__)


def init_tf_rng(spec):
    if spec is None:
        return

    seed = util.random.rng_seed(spec)
    seed, = seed.generate_state(1)
    tf.random.set_seed(seed)


def make_graph(rng_spec=None):
    "Construct a TensorFlow graph (with an optional random seed)"
    rng = util.rng(rng_spec)
    graph = tf.Graph()
    graph.seed = rng.integers(2**31 - 1)
    _log.info('using effective random seed %s (from %s)', graph.seed, rng_spec)
    return graph
