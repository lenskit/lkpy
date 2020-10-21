from logging import getLogger
from lenskit import util

try:
    import tensorflow as tf
except ImportError:
    tf = None

_log = getLogger(__name__)


def have_usable_tensorflow():
    if tf is None:
        return False
    elif tf.__version__ < '2':
        return False
    else:
        return True


def check_tensorflow():
    if tf is None:
        raise ImportError('tensorflow')
    elif not have_usable_tensorflow():
        raise RuntimeError('TensorFlow not usable, too old?')


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
