"""
Support for sharing objects (e.g. trained models) between processes.
"""

from abc import ABCMeta, abstractmethod
from pathlib import Path
import logging
import uuid

import pandas as pd
import numpy as np
import pyarrow as pa

_logger = logging.getLogger(__package__)


class ObjectRepo(metaclass=ABCMeta):
    """
    Interface for shared data repositories.
    """

    @abstractmethod
    def share(self, object):
        """
        Publish an object to the repository.  The object itself will usually not be shared directly,
        but it will be copied into shared storage so that :py:meth:`resolve` will be return a copy
        that is shared as efficiently as feasible.

        Args:
            object(shareable):
                the object to be shared.

        Returns:
            the object's key in the repository.
        """
        raise NotImplemented()

    @abstractmethod
    def resolve(self, key):
        """
        Resolve a key to an object from the repository.

        Args:
            key: the object key to resolve or retrieve.

        Returns:
            a reference to or copy of the shared object.  Client code must not try to modify
            this object.
        """


class FileRepo(ObjectRepo):
    """
    Share objects by serializing them to Arrow files in a directory.  Keys are UUIDs.
    """

    def __init__(self, dir, engine=None):
        self.dir = Path(dir)
        self._pq_args = {}
        if engine is not None:
            self._pq_args['engine'] = engine

        if not self.dir.exists():
            raise ValueError('directory {} does not exist', dir)

    def share(self, object):
        key = uuid.uuid4()
        _logger.debug('sharing object %s to %s', repr(object), key)

        fn = self.dir / '{}.parquet'.format(key)
        object.to_parquet(fn)
        _logger.info('saved %s to %s (%d bytes)', repr(object), key,
                     fn.stat().st_size)

        return key

    def resolve(self, key):
        fn = self.dir / '{}.parquet'.format(key)
        if not fn.exists():
            raise KeyError(key)

        _logger.info('reading %s to %s (%d bytes)', repr(object), key,
                     fn.stat().st_size)

        return pd.read_parquet(fn)
