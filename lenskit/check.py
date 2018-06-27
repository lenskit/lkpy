"""
Utility functions for precondition checks.
"""

import warnings


def check_value(expr, msg, *args, warn=False):
    if not expr:
        if warn:
            warnings.warn(msg.format(*args))
        else:
            raise ValueError(msg.format(*args))
