import logging

from lenskit.sharing import persist, persist_binpickle
from lenskit.util.parallel import run_sp
from lenskit.util import Stopwatch

_log = logging.getLogger(__name__)


def _train_and_save(algo, file, ratings, kwargs):
    "Worker for subprocess model training"
    _log.info('training %s on %d ratings', algo, len(ratings))
    timer = Stopwatch()
    algo.fit(ratings, **kwargs)
    _log.info('trained %s in %s', algo, timer)
    if file is None:
        return persist_binpickle(algo).transfer()
    else:
        return persist_binpickle(algo, file=file).transfer()


def train_isolated(algo, ratings, *, file=None, **kwargs):
    """
    Train an algorithm in a subprocess to isolate the training process.  This
    function spawns a subprocess (in the same way that LensKit's multiprocessing
    support does), calls :meth:`lenskit.algorithms.Algorithm.fit` on it, and
    serializes the result for shared-memory use.

    Training the algorithm in a single-purpose subprocess makes sure that any
    training resources, such as TensorFlow sessions, are cleaned up by virtue
    of the process terminating when model training is completed.  It can also
    reduce memory use, because the original trained model and the shared memory
    version are not in memory at the same time.  While the batch functions use
    shared memory to reduce memory overhead for parallel processing, naive use
    of these functions will still have 2 copies of the model in memory, the
    shared one and the original, because the sharing process does not tear down
    the original model.  Training in a subprocess solves this problem elegantly.

    Args:
        algo(lenskit.algorithms.Algorithm):
            The algorithm to train.
        ratings(pandas.DataFrame):
            The rating data.
        file(str or pathlib.Path or None):
            The file in which to save the trained model.  If ``None``, uses a
            default file path or shared memory.
        kwargs(dict):
            Additional named parameters to :meth:`lenskit.algorithms.Algorithm.fit`.

    Returns:
        lenskit.sharing.PersistedObject:
            The saved model object.  This is the owner, so it needs to be closed
            when finished to free resources.
    """

    return run_sp(_train_and_save, algo, file, ratings, kwargs)
