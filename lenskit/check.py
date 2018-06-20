"""
Utility functions for precondition checks.
"""


def check_value(expr, msg, *args):
    if not expr:
        raise ValueError(msg.format(*args))
