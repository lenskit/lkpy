"""
File utilities
"""

import atexit
import os
import logging
import pathlib

__all__ = ['delete_sometime', 'fspath', 'norm_path']

_log = logging.getLogger(__name__)
__os_fp = getattr(os, 'fspath', None)

_removable_files = []


@atexit.register
def _cleanup_files():
    for f in _removable_files:
        if f.exists():
            try:
                _log.debug('deleting %s', f)
                f.unlink()
            except PermissionError as e:
                _log.warn('could not delete %s: %s', f, e)
            except FileNotFoundError:
                _log.debug('%s not found, delete race?')


def delete_sometime(f):
    """
    Delete a file. If it is not possible now (e.g. it is open on Windows), arrange for it
    to be deleted at process exit.
    """
    if f is not None and f.exists():
        try:
            f.unlink()
        except PermissionError:
            _log.debug('cannot delete %s, scheduling', f)
            _removable_files.append(f)


def fspath(path):
    "Backport of :py:func:`os.fspath` function for Python 3.5."
    if __os_fp:
        return __os_fp(path)
    else:
        return str(path)


def norm_path(path):
    """
    Convert a path into a :cls:`pathlib.Path`, in a Python 3.5-compatible way.
    """
    if isinstance(path, pathlib.Path):
        return path
    elif hasattr(path, '__fspath__'):
        return pathlib.Path(path.__fspath__())
    elif isinstance(path, str):
        return pathlib.Path(str)
    else:
        raise ValueError('invalid path: ' + repr(path))
