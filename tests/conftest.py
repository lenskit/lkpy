import logging
from pytest import fixture

from lenskit.util import test as lktu

logging.getLogger('numba').setLevel(logging.INFO)

_log = logging.getLogger('lenskit.tests')


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
