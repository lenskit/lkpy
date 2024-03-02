# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Debugging utility code.  Also runnable as a Python command.

Usage:
    lenskit.util.debug [options] --blas-info
    lenskit.util.debug [options] --numba-info
    lenskit.util.debug [options] --check-env

Options:
    --verbose
        Turn on verbose logging
"""

import sys
import logging
from typing import Optional
from dataclasses import dataclass
import numba
import threadpoolctl

from .parallel import is_worker

_log = logging.getLogger(__name__)
_already_checked = False


@numba.njit(parallel=True)
def _par_test(n):
    "Test function to activate Numba parallel config."
    x = 0
    for i in numba.prange(n):
        x += n

    return x


@dataclass
class BlasInfo:
    impl: str
    threading: Optional[str]
    threads: Optional[int]
    version: str


@dataclass
class NumbaInfo:
    threading: str
    threads: int


def blas_info():
    pools = threadpoolctl.threadpool_info()
    blas = None
    for pool in pools:
        if pool["user_api"] != "blas":
            continue

        if blas is not None:
            _log.warning("found multiple BLAS layers, using first")
            _log.info("later layer is: %s", pool)
            continue

        blas = BlasInfo(
            pool["internal_api"],
            pool.get("threading_layer", None),
            pool.get("num_threads", None),
            pool["version"],
        )

    return blas


def numba_info():
    x = _par_test(100)
    _log.debug("sum: %d", x)

    try:
        layer = numba.threading_layer()
    except ValueError:
        _log.info("Numba threading not initialized")
        return None
    _log.info("numba threading layer: %s", layer)
    nth = numba.get_num_threads()
    return NumbaInfo(layer, nth)


def check_env():
    """
    Check the runtime environment for potential performance or stability problems.
    """
    global _already_checked
    problems = 0
    if _already_checked or is_worker():
        return

    try:
        blas = blas_info()
        numba = numba_info()
    except Exception as e:
        _log.error("error inspecting runtime environment: %s", e)
        _already_checked = True
        return

    if numba is None:
        _log.warning("Numba JIT seems to be disabled - this will hurt performance")
        _already_checked = True
        return

    if blas is None:
        _log.warning("threadpoolctl could not find your BLAS")
        _already_checked = True
        return

    _log.info("Using BLAS %s", blas.impl)

    if numba.threading != "tbb":
        _log.info("Numba is using threading layer %s - consider TBB", numba.threading)

    if numba.threading == "tbb" and blas.threading == "tbb":
        _log.info("Numba and BLAS both using TBB - good")

    if numba.threading == "tbb" and blas.impl == "mkl" and blas.threading != "tbb":
        _log.warning("Numba using TBB but MKL is using %s", blas.threading)
        _log.info("Set MKL_THREADING_LAYER=tbb for improved performance")
        problems += 1

    if blas.threads and blas.threads > 1 and numba.threads > 1:
        # TODO make this be fine in OpenMP configurations
        _log.warning("BLAS using multiple threads - can cause oversubscription")
        _log.info("See https://mde.one/lkpy-blas for information on tuning BLAS for LensKit")
        problems += 1

    if problems:
        _log.warning("found %d potential runtime problems - see https://boi.st/lkpy-perf", problems)

    _already_checked = True
    return problems


def print_blas_info():
    blas = blas_info()
    print(blas)


def print_numba_info():
    numba = numba_info()
    print(numba)


def main():
    from docopt import docopt

    opts = docopt(__doc__)
    level = logging.DEBUG if opts["--verbose"] else logging.INFO
    logging.basicConfig(level=level, stream=sys.stderr, format="%(levelname)s %(name)s %(message)s")
    logging.getLogger("numba").setLevel(logging.INFO)

    if opts["--blas-info"]:
        print_blas_info()
    if opts["--numba-info"]:
        print_numba_info()
    if opts["--check-env"]:
        check_env()


if __name__ == "__main__":
    _log = logging.getLogger("lenskit.util.debug")
    main()
