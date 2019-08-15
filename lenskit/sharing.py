"""
Support for sharing and saving models and data structures.
"""

from contextlib import contextmanager

__save_mode = 'save'


@contextmanager
def sharing_mode():
    """
    Context manager to tell models that pickling will be used for cross-process
    sharing, not model persistence.
    """
    global __save_mode
    old = __save_mode
    __save_mode = 'share'
    try:
        yield
    finally:
        __save_mode = old


def in_share_context():
    """
    Query whether sharing mode is active.  If ``True``, we are currently in a
    :fun:`sharing_mode` context, which means model pickling will be used for
    cross-process sharing.
    """
    return __save_mode == 'share'
