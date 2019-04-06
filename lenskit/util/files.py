"""
File utilities
"""

import atexit
import os
import logging

_log = logging.getLogger(__name__)
__os_fp = getattr(os, 'fspath', None)


def fspath(path):
    "Backport of :py:func:`os.fspath` function for Python 3.5."
    if __os_fp:
        return __os_fp(path)
    else:
        return str(path)
