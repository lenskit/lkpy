import os
import os.path
import logging
import tempfile
import pathlib
from joblib import Parallel, delayed, dump, load

import pandas as pd

from .. import util

_logger = logging.getLogger(__name__)
_rec_context = None


def _predict_user(algo, user, udf):
    watch = util.Stopwatch()
    res = algo.predict_for_user(user, udf['item'])
    res = pd.DataFrame({'user': user, 'item': res.index, 'prediction': res.values})
    _logger.debug('%s produced %f/%d predictions for %s in %s',
                  algo, res.prediction.notna().sum(), len(udf), user, watch)
    return res


def predict(algo, pairs, *, nprocs=None):
    """
    Generate predictions for user-item pairs.  The provided algorithm should be a
    :py:class:`algorithms.Predictor` or a function of two arguments: the user ID and
    a list of item IDs. It should return a dictionary or a :py:class:`pandas.Series`
    mapping item IDs to predictions.

    To use this function, provide a pre-fit algorithm::

        >>> from lenskit.algorithms.basic import Bias
        >>> from lenskit.metrics.predict import rmse
        >>> ratings = util.load_ml_ratings()
        >>> bias = Bias()
        >>> bias.fit(ratings[:-1000])
        <lenskit.algorithms.basic.Bias object at ...>
        >>> preds = predict(bias, ratings[-1000:])
        >>> preds.head()
               user  item  rating   timestamp  prediction
        99004   664  8361     3.0  1393891425    3.288286
        99005   664  8528     3.5  1393891047    3.559119
        99006   664  8529     4.0  1393891173    3.573008
        99007   664  8636     4.0  1393891175    3.846268
        99008   664  8641     4.5  1393890852    3.710635
        >>> rmse(preds['prediction'], preds['rating'])
        0.8326992222...

    Args:
        algo(lenskit.algorithms.Predictor):
            A rating predictor function or algorithm.
        pairs(pandas.DataFrame):
            A data frame of (``user``, ``item``) pairs to predict for. If this frame also
            contains a ``rating`` column, it will be included in the result.
        nprocs(int):
            The number of processes to use for parallel batch prediction.  Passed as
            ``n_jobs`` to :cls:`joblib.Parallel`.  The default, ``None``, will make
            the process sequential _unless_ called inside the :func:`joblib.parallel_backend`
            context manager.

    Returns:
        pandas.DataFrame:
            a frame with columns ``user``, ``item``, and ``prediction`` containing
            the prediction results. If ``pairs`` contains a `rating` column, this
            result will also contain a `rating` column.
    """

    loop = Parallel(n_jobs=nprocs)

    path = None
    try:
        if loop._effective_n_jobs() > 1:
            fd, path = tempfile.mkstemp(prefix='lkpy-predict', suffix='.pkl')
            path = pathlib.Path(path)
            os.close(fd)
            _logger.debug('pre-serializing algorithm %s to %s', algo, path)
            dump(algo, path)
            algo = load(path, mmap_mode='r')

        results = loop(delayed(_predict_user)(algo, user, udf)
                       for (user, udf) in pairs.groupby('user'))
        del algo

        results = pd.concat(results, ignore_index=True)
    finally:
        util.delete_sometime(path)

    if 'rating' in pairs:
        return pairs.join(results.set_index(['user', 'item']), on=('user', 'item'))
    return results
