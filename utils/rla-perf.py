"""
Little script to test RLA performance.

Usage:
    rla-perf.py (--prepare|--measure) [options]

Options:
    --prepare
        Prepare for running.
    --measure
        Measure the results.
    -d DATASET
        Use dataset DATASET [default: ml-20m]
    --verbose
        Enable debug logging.
"""

import sys
import logging

import tqdm
from docopt import docopt
import pandas as pd

from lenskit.datasets import MovieLens
from lenskit.util import Stopwatch
from lenskit.batch import recommend
from lenskit.crossfold import sample_users, SampleN
from lenskit.algorithms.basic import Popular
from lenskit.algorithms import Recommender
from lenskit.algorithms.als import ImplicitMF
from lenskit.topn import RecListAnalysis, ndcg, recip_rank

_log = logging.getLogger('rla-perf')


def do_prepare(opts):
    name = opts['-d']
    ml = MovieLens(f'data/{name}')

    train, test = next(sample_users(ml.ratings, 1, 10000, SampleN(5)))

    test.to_parquet(f'data/{name}-test.parquet', index=False)

    _log.info('getting popular recs')
    pop = Popular()
    pop.fit(train)
    pop_recs = recommend(pop, test['user'].unique(), 100)

    _log.info('getting ALS recs')
    als = ImplicitMF(20, iterations=10)
    als = Recommender.adapt(als)
    als.fit(train.drop(columns=['rating']))
    als_recs = recommend(als, test['user'].unique(), 100)

    _log.info('merging recs')
    recs = pd.concat({
        'Popular': pop_recs,
        'ALS': als_recs
    }, names=['Algorithm'])
    recs.reset_index('Algorithm', inplace=True)
    recs.to_parquet(f'data/{name}-recs.parquet', index=False)


def do_measure(opts):
    name = opts['-d']

    _log.info('reading data %s', name)
    test = pd.read_parquet(f'data/{name}-test.parquet')
    recs = pd.read_parquet(f'data/{name}-recs.parquet')

    _log.info('setting up analysis')
    rla = RecListAnalysis()
    rla.add_metric(ndcg)
    rla.add_metric(recip_rank)

    timer = Stopwatch()
    results = rla.compute(recs, test, include_missing=True)
    _log.info('analyzed in %s', timer)

    results = results.fillna(0)
    a_res = results.groupby('Algorithm').mean()
    a_res['count'] = results.groupby('Algorithm')['nrecs'].count()
    _log.info('finished')
    print(a_res)
    print(results.groupby('Algorithm')['recip_rank'].describe())


if __name__ == '__main__':
    opts = docopt(__doc__)
    level = logging.DEBUG if opts['--verbose'] else logging.INFO
    format = '%(relativeCreated)6d: %(levelname)s %(name)s %(message)s'
    logging.basicConfig(stream=sys.stderr, level=level, format=format)
    tqdm.tqdm.pandas()

    if opts['--prepare']:
        do_prepare(opts)
    elif opts['--measure']:
        do_measure(opts)
    else:
        _log.error('no operation specified')
        sys.exit(2)
