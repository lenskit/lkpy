import logging
from pytest import fixture

import numpy as np

logging.getLogger('numba').setLevel(logging.INFO)

_log = logging.getLogger('lenskit.tests')


@fixture(scope='session')
def rng():
    if hasattr(np.random, 'default_rng'):
        return np.random.default_rng()
    else:
        return np.random.RandomState()


@fixture(autouse=True)
def log_test(request):
    _log.info('running test %s:%s', request.module.__name__, request.function.__name__)


def pytest_collection_modifyitems(items):
    # add 'slow' to all 'eval' tests
    for item in items:
        evm = item.get_closest_marker('eval')
        slm = item.get_closest_marker('slow')
        if evm is not None and slm is None:
            _log.debug('adding slow mark to %s', item)
            item.add_marker('slow')
