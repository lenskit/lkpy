"""
Logging utilities.
"""

import sys
import logging

_log = logging.getLogger(__name__)
_lts_initialized = False
_ltn_initialized = False


class LowPassFilter:
    def filter(record):
        return record.levelno < logging.WARNING


def log_to_stderr(level=logging.INFO):
    """
    Set up the logging infrastructure to show log output on ``sys.stderr``, where it will
    appear in the IPython message log.
    """
    global _lts_initialized
    if _lts_initialized:
        _log.info('log already initialized')

    h = logging.StreamHandler(sys.stderr)
    f = logging.Formatter('[%(levelname)7s] %(name)s %(message)s')
    h.setFormatter(f)
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(level)

    _log.info('stderr logging configured')
    _lts_initialized = True


def log_to_notebook(level=logging.INFO):
    """
    Set up the logging infrastructure to show log output in the Jupyter notebook.
    """
    global _ltn_initialized
    if _ltn_initialized:
        _log.info('log already initialized')

    h = logging.StreamHandler(sys.stderr)
    f = logging.Formatter('[%(levelname)7s] %(name)s %(message)s')
    h.setFormatter(f)
    h.setLevel(logging.WARNING)

    oh = logging.StreamHandler(sys.stdout)
    oh.setFormatter(f)
    oh.addFilter(LowPassFilter)

    root = logging.getLogger()
    root.addHandler(h)
    root.addHandler(oh)
    root.setLevel(level)

    _log.info('notebook logging configured')
    _ltn_initialized = True
