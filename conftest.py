import logging
from pytest import fixture

import numpy as np

from lenskit import util

logging.getLogger('numba').setLevel(logging.INFO)

_log = logging.getLogger('lenskit.tests')


@fixture
def rng():
    return util.rng(42)


@fixture
def legacy_rng():
    return util.rng(42, legacy_rng=True)


@fixture(autouse=True)
def init_rng(request):
    util.init_rng(42)


@fixture(autouse=True)
def log_test(request):
    modname = request.module.__name__ if request.module else '<unknown>'
    funcname = request.function.__name__ if request.function else '<unknown>'
    _log.info('running test %s:%s', modname, funcname)


@fixture(autouse=True, scope='session')
def carbon(request):
    try:
        from codecarbon import EmissionsTracker
    except ImportError:
        yield True  # we do nothing
        return

    tracker = EmissionsTracker("lkpy-tests", 5)
    tracker.start()
    try:
        yield True
    finally:
        emissions = tracker.stop()
        _log.info('test suite used %.3f kgCO2eq', emissions)


def pytest_collection_modifyitems(items):
    # add 'slow' to all 'eval' tests
    for item in items:
        evm = item.get_closest_marker('eval')
        slm = item.get_closest_marker('slow')
        if evm is not None and slm is None:
            _log.debug('adding slow mark to %s', item)
            item.add_marker('slow')
