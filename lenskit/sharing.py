"""
Support for sharing data between processes to enable multi-process
evaluation operations more easily.
"""

from abc import ABCMeta, abstractmethod, abstractclassmethod
import logging
from multiprocessing import sharedctypes as mpctypes

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)


class Shareable(metaclass=ABCMeta):
    """
    Interface for objects and classes that can share themselves.
    """

    @abstractmethod
    def share_publish(self):
        raise NotImplementedError()

    @abstractclassmethod
    def share_resolve(self, key):
        raise NotImplementedError()


class ShareHelper(metaclass=ABCMeta):
    """
    Interfaces for classes that can help publish other objects.  This is useful for
    algorithms to be able to publish their models.
    """

    @abstractmethod
    def share_publish(self, model):
        raise NotImplementedError()

    @abstractmethod
    def share_resolve(self, key):
        raise NotImplementedError()


def put_series(series):
    i_k = put_index(series.index)
    v_k = put_array(series.values)
    return (series.name, v_k, i_k)


def get_series(key):
    name, v_k, i_k = key
    index = get_index(i_k)
    values = get_array(v_k)
    series = pd.Series(values, index=index, name=name)
    return series


def put_index(index):
    v_k = put_array(index.values)
    return (index.name, v_k)


def get_index(key):
    name, v_k = key
    values = get_array(v_k)
    return pd.Index(values, name=name)


def put_array(array):
    _logger.debug('sharing object of type %s and shape %s (size=%d)',
                  array.dtype.str, array.shape, array.size)
    code = np.ctypeslib._typecodes[array.dtype.str]
    shared = mpctypes.Array(code, array.size, lock=False)
    shape = array.shape
    nda = np.ctypeslib.as_array(shared)
    nda[:] = array.reshape(array.size)

    return (shape, shared)


def get_array(key):
    shape, shared = key
    nda = np.ctypeslib.as_array(shared)
    nda = nda.reshape(shape)
    return nda


def publish(obj, helper=None):
    """
    Publish an object for sharing.

    Args:
        obj: the object to share.
        helper:
            a helper to try to use.  If this is not an instance of :py:class:`ShareHelper`, it is
            ignored.
    """

    if isinstance(helper, ShareHelper):
        return helper.share_publish(obj)
    elif isinstance(obj, Shareable):
        return (obj.share_publish(), obj.__class__)
    else:
        raise TypeError('{} is not shareable and no suitable helper provided'.format(obj))


def resolve(key, helper):
    """
    Resolve an object for sharing.

    Args:
        obj: the object to share.
        helper(ShareHelper):
            a helper to try to use.  If this is not an instance of :py:class:`ShareHelper`, it is
            ignored.
    """

    if not isinstance(helper, ShareHelper):
        key, helper = key
    return helper.share_resolve(key)


def is_sharable(obj, helper=None):
    """
    Test whether we have sufficient APIs to share an object.

    Args:
        obj: the object to share
        helper(ShareHelper):
            a helper to try to use.  If this is not an instance of :py:class:`ShareHelper`, it is
            ignored.
    """

    return isinstance(obj, Shareable) or isinstance(helper, ShareHelper)
