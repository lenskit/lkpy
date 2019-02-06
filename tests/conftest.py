import logging
from pytest import fixture

_log = logging.getLogger('lenskit.tests')


@fixture(autouse=True)
def log_test(request):
    _log.info('running test %s:%s', request.module.__name__, request.function.__name__)
