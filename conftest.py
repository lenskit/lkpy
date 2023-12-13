import logging
from pytest import fixture

from seedbank import numpy_rng, initialize

logging.getLogger("numba").setLevel(logging.INFO)

_log = logging.getLogger("lenskit.tests")


@fixture
def rng():
    return numpy_rng(42)


@fixture(autouse=True)
def init_rng(request):
    initialize(42)


@fixture(autouse=True)
def log_test(request):
    modname = request.module.__name__ if request.module else "<unknown>"
    funcname = request.function.__name__ if request.function else "<unknown>"
    _log.info("running test %s:%s", modname, funcname)


def pytest_collection_modifyitems(items):
    # add 'slow' to all 'eval' tests
    for item in items:
        evm = item.get_closest_marker("eval")
        slm = item.get_closest_marker("slow")
        if evm is not None and slm is None:
            _log.debug("adding slow mark to %s", item)
            item.add_marker("slow")
