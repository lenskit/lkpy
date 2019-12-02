"""
File utilities
"""

import atexit
import os
import tempfile
import logging
import pathlib

__all__ = ['delete_sometime', 'scratch_dir']

_log = logging.getLogger(__name__)

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


def scratch_dir(default=True, joblib=False):
    """
    Get the configured temporary directory.  Looks for configuration in the following
    places:

    1. The environment variable ``LENSKIT_TEMP_DIR``.
    2. If ``joblib`` is ``True``, the environment variable ``JOBLIB_TEMP_FOLDER``.
    3. If ``default`` is ``True``, the result of :fun:`tempfile.gettempdir`.

    Args:
        default(bool): whether to look in the Python default locations.
        joblib(bool): whether to consult the Joblib configuration directory.
    """
    path = os.environ.get('LENSKIT_TEMP_DIR', None)
    if joblib and not path:
        path = os.environ.get('JOBLIB_TEMP_FOLDER', None)
    if default and not path:
        path = tempfile.gettempdir()
    return path
