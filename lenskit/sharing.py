"""
Support for sharing objects (e.g. trained models) between processes.
"""

from abc import ABCMeta, abstractmethod
from pathlib import Path
import logging
import uuid
import json

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as parq

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

    def __init__(self, dir):
        self.dir = Path(dir)

        if not self.dir.exists():
            raise ValueError('directory {} does not exist', dir)

    def share(self, object):
        key = uuid.uuid4()
        _logger.debug('sharing object to %s', key)

        fn = self.dir / '{}.parquet'.format(key)
        tbl, schema = self._to_table(object)
        pqf = parq.ParquetWriter(fn, schema)
        try:
            pqf.write_table(tbl)
        finally:
            pqf.close()

        _logger.info('saved %s (%d bytes)', key, fn.stat().st_size)

        return key

    def resolve(self, key):
        fn = self.dir / '{}.parquet'.format(key)
        if not fn.exists():
            raise KeyError(key)

        _logger.info('reading %s to %s (%d bytes)', repr(object), key,
                     fn.stat().st_size)

        pqf = parq.ParquetFile(fn)

        tbl = pqf.read()

        return self._from_tbl(tbl)

    def _to_table(self, obj):
        if isinstance(obj, pd.DataFrame):
            tbl = pa.Table.from_pandas(obj)
            return tbl, tbl.schema
        elif isinstance(obj, pd.Series):
            name = obj.name
            if name is None:
                name = 'data'
            df = pd.DataFrame({name: obj})
            tbl = pa.Table.from_pandas(df)
            meta = dict(tbl.schema.metadata)
            meta[b'lkpy'] = json.dumps({'type': 'series', 'name': name}).encode()
            schema = tbl.schema.add_metadata(meta)
            return tbl, schema
        else:
            raise ValueError('unserializable type {}'.format(type(obj)))

    def _from_tbl(self, tbl):
        meta = tbl.schema.metadata
        lk_meta = meta.get(b'lkpy')
        if lk_meta is not None:
            lk_meta = json.loads(lk_meta.decode())
        else:
            lk_meta = {}

        if lk_meta.get('type') == 'series':
            _logger.debug('decoding as Pandas series')
            return tbl.to_pandas()[lk_meta['name']]
        else:
            return tbl.to_pandas()
