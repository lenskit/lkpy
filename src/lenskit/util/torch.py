"""
PyTorch utility functions.
"""

import functools

import torch


def inference_mode(func):
    """
    Function decorator that puts PyTorch in inference mode.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with torch.inference_mode():
            return func(*args, **kwargs)

    return wrapper
