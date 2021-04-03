"""
Debugging utility code.  Also runnable as a Python command.

Usage:
    lenskit.util.debug [options] --libraries
    lenskit.util.debug [options] --blas-info
    lenskit.util.debug [options] --numba-info

Options:
    --verbose
        Turn on verbose logging
"""

import sys
import logging
import ctypes
from typing import Optional
from dataclasses import dataclass
import numba
import psutil
import numpy as np

_log = logging.getLogger(__name__)


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


def _guess_layer():
    proc = psutil.Process()
    layer = None
    if not hasattr(proc, 'memory_maps'):
        return None

    _log.debug('scanning process memory maps for threading layers')
    for mm in proc.memory_maps():
        if 'mkl_intel_thread' in mm.path:
            _log.debug('found library %s linked', mm.path)
            if layer:
                _log.warn('multiple threading layers detected')
            layer = 'intel'
        elif 'mkl_tbb_thread' in mm.path:
            _log.debug('found library %s linked', mm.path)
            if layer:
                _log.warn('multiple threading layers detected')
            layer = 'tbb'
        elif 'mkl_gnu_thread' in mm.path:
            _log.debug('found library %s linked', mm.path)
            if layer:
                _log.warn('multiple threading layers detected')
            layer = 'gnu'

    return layer


def guess_blas_unix():
    _log.info('opening self DLL')
    dll = ctypes.CDLL(None)

    _log.debug('checking for MKL')
    try:
        mkl_vstr = dll.mkl_get_version_string
        mkl_vbuf = ctypes.create_string_buffer(256)
        mkl_vstr(mkl_vbuf, 256)
        version = mkl_vbuf.value.decode().strip()
        _log.debug('version %s', version)

        mkl_mth = dll.mkl_get_max_threads
        mkl_mth.restype = ctypes.c_int
        threads = mkl_mth()

        layer = _guess_layer()

        return BlasInfo('mkl', layer, threads, version)
    except AttributeError:
        pass  # no MKL

    _log.debug('checking for OpenBLAS')
    np_dll = ctypes.CDLL(np.core._multiarray_umath.__file__)
    try:
        openblas_vstr = np_dll.openblas_get_config
        openblas_vstr.restype = ctypes.c_char_p
        version = openblas_vstr().decode()
        _log.debug('version %s', version)

        openblas_th = np_dll.openblas_get_num_threads
        openblas_th.restype = ctypes.c_int
        threads = openblas_th()
        _log.debug('threads %d', threads)

        return BlasInfo('openblas', None, threads, version)
    except AttributeError as e:
        _log.info('OpenBLAS error: %s', e)

    return BlasInfo(None, None, None, 'unknown')


def _win_search_dlls(**kwargs):
    for key, dll in kwargs.items():
        try:
            _log.debug('looking for %s', dll)
            getattr(ctypes.cdll, dll)
            return key
        except FileNotFoundError:
            pass


def guess_blas_windows():
    try:
        blas_dll = ctypes.cdll.libblas
        _log.debug('loaded MKL dll %s', blas_dll)
    except FileNotFoundError as e:
        _log.error('could not load LIBBLAS')
        raise e

    _log.debug('checking for MKL')
    try:
        mkl_vstr = blas_dll.mkl_get_version_string
        mkl_vbuf = ctypes.create_string_buffer(256)
        mkl_vstr(mkl_vbuf, 256)
        version = mkl_vbuf.value.decode().strip()
        _log.debug('version %s', version)

        mkl_mth = blas_dll.mkl_get_max_threads
        mkl_mth.restype = ctypes.c_int
        threads = mkl_mth()

        layer = _guess_layer()

        return BlasInfo('mkl', layer, threads, version)
    except AttributeError:
        pass  # no MKL

    _log.debug('checking for OpenBLAS')
    try:
        openblas_vstr = blas_dll.openblas_get_config
        openblas_vstr.restype = ctypes.c_char_p
        version = openblas_vstr().decode()

        openblas_th = blas_dll.openblas_get_num_threads
        openblas_th.restype = ctypes.c_int
        threads = openblas_th()
        _log.debug('threads %d', threads)

        return BlasInfo('openblas', None, threads, version)
    except AttributeError as e:
        _log.info('OpenBLAS error: %s', e)

    return BlasInfo(None, None, None, 'unknown')


def blas_info():
    if sys.platform == 'win32':
        return guess_blas_windows()
    else:
        return guess_blas_unix()


def numba_info():
    x = _par_test(100)
    _log.debug('sum: %d', x)

    try:
        layer = numba.threading_layer()
    except ValueError:
        _log.info('Numba threading not initialized')
        return None
    _log.info('numba threading layer: %s', layer)
    nth = numba.get_num_threads()
    return NumbaInfo(layer, nth)


def print_libraries():
    p = psutil.Process()

    _log.info('printing process libraries')
    for map in p.memory_maps():
        print(map.path)


def print_blas_info():
    blas = blas_info()
    print(blas)


def print_numba_info():
    numba = numba_info()
    print(numba)


def main():
    from docopt import docopt
    opts = docopt(__doc__)
    level = logging.DEBUG if opts['--verbose'] else logging.INFO
    logging.basicConfig(level=level, stream=sys.stderr, format='%(levelname)s %(name)s %(message)s')

    if opts['--libraries']:
        print_libraries()
    if opts['--blas-info']:
        print_blas_info()
    if opts['--numba-info']:
        print_numba_info()


if __name__ == '__main__':
    _log = logging.getLogger('lenskit.util.debug')
    main()
