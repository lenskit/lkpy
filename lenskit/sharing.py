"""
Support for sharing objects (e.g. trained models) between processes.
"""

from abc import ABCMeta, abstractmethod
from pathlib import Path
import logging
import uuid
import json
import tempfile

import pandas as pd
import numpy as np
import scipy as sp
import pyarrow as pa
import pyarrow.parquet as parq

_logger = logging.getLogger(__package__)

try:
    from pyarrow import plasma
except ImportError:
    _logger.info('plasma not available')


def _to_table(obj):
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
    elif isinstance(obj, np.ndarray):
        arr = pa.Array.from_pandas(obj)
        meta = {b'lkpy': json.dumps({'type': 'array'}).encode()}
        tbl = pa.Table.from_arrays([arr], ['array'], metadata=meta)
        return tbl, tbl.schema
    elif sp.sparse.isspmatrix_coo(obj):
        data = pa.Array.from_pandas(obj.data)
        _logger.debug('obj type: %s', obj.row.dtype)
        rows = pa.Array.from_pandas(obj.row.astype(np.int))
        cols = pa.Array.from_pandas(obj.col.astype(np.int))
        info = {'type': 'spmatrix', 'layout': 'coo'}
        info['shape'] = {'rows': obj.shape[0], 'cols': obj.shape[1]}
        meta = {b'lkpy': json.dumps(info).encode()}
        tbl = pa.Table.from_arrays([rows, cols, data], ['row', 'col', 'data'], metadata=meta)
        return tbl, tbl.schema
    else:
        raise ValueError('unserializable type {}'.format(type(obj)))


def _from_tbl(tbl):
    meta = tbl.schema.metadata
    lk_meta = meta.get(b'lkpy')
    if lk_meta is not None:
        lk_meta = json.loads(lk_meta.decode())
    else:
        lk_meta = {}

    if lk_meta.get('type') == 'series':
        _logger.debug('decoding as Pandas series')
        return tbl.to_pandas()[lk_meta['name']]
    elif lk_meta.get('type') == 'array':
        _logger.debug('decoding as NumPy array')
        col = tbl.column(0)
        _logger.debug('column has %d rows in %d chunks', len(col), col.data.num_chunks)
        arr = np.concatenate([chunk.to_pandas() for chunk in col.data.iterchunks()])
        _logger.debug('decoded array of shape %s and type %s', arr.shape, arr.dtype)
        return arr
    elif lk_meta.get('type') == 'spmatrix':
        shape = (lk_meta['shape']['rows'], lk_meta['shape']['cols'])
        _logger.debug('decoding sparse matrix with shape %s', shape)
        if lk_meta['layout'] != 'coo':
            raise ValueError('unknown sparse matrix layout {}'.format(lk_meta['layout']))
        row = np.concatenate([chunk.to_pandas().astype(np.int32)
                              for chunk in tbl.column(0).data.iterchunks()])
        col = np.concatenate([chunk.to_pandas().astype(np.int32)
                              for chunk in tbl.column(1).data.iterchunks()])
        data = np.concatenate([chunk.to_pandas() for chunk in tbl.column(2).data.iterchunks()])
        return sp.sparse.coo_matrix((data, (row, col)), shape=shape)
    else:
        return tbl.to_pandas()


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
        raise NotImplemented

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
        raise NotImplemented

    def close(self):
        """
        Close down the repository.
        """
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class FileRepo(ObjectRepo):
    """
    Share objects by serializing them to Arrow files in a directory.  Keys are UUIDs.
    """

    def __init__(self, dir=None):
        if dir is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix='lkpy-repo')
            self.dir = Path(self._tmpdir.name)
        else:
            self.dir = Path(dir)
            self._tmpdir = None

        if not self.dir.exists():
            raise ValueError('directory {} does not exist', dir)

    def close(self):
        if self._tmpdir is not None:
            self._tmpdir.cleanup()

    def share(self, object):
        key = uuid.uuid4()
        _logger.debug('sharing object to %s', key)

        fn = self.dir / '{}.parquet'.format(key)
        tbl, schema = _to_table(object)
        _logger.debug('using schema %s', schema)
        pqf = parq.ParquetWriter(str(fn), schema)
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

        _logger.info('reading %s from %s (%d bytes)', repr(object), key,
                     fn.stat().st_size)

        pqf = parq.ParquetFile(str(fn))

        tbl = pqf.read()

        return _from_tbl(tbl)


class PlasmaRepo(ObjectRepo):
    """
    Share objects by serializing them to Arrow files in a directory.  Keys are UUIDs.
    """

    def __init__(self, socket, manager="", release_delay=0):
        self.client = plasma.connect(socket, manager, release_delay)

    def share(self, object):
        id = plasma.ObjectID.from_random()
        tbl, schema = _to_table(object)

        _logger.debug('finding size to write to %s', id)
        mock = pa.MockOutputStream()
        sw = pa.RecordBatchStreamWriter(mock, schema)
        sw.write_table(tbl)
        sw.close()
        size = mock.size()
        _logger.debug('writing %d bytes to %s', size, id)

        buf = self.client.create(id, size)
        stream = pa.FixedSizeBufferWriter(buf)
        sw = pa.RecordBatchStreamWriter(stream, schema)
        sw.write_table(tbl)
        sw.close()
        self.client.seal(id)

        _logger.info('shared object to %s', id)
        return id

    def resolve(self, key):
        _logger.info('resolving object %s', key)
        [data] = self.client.get_buffers([key])
        buf = pa.BufferReader(data)
        _logger.debug('reading object of size %d', buf.size())
        reader = pa.RecordBatchStreamReader(buf)
        tbl = reader.read_all()
        _logger.debug('loaded table with schema %s', tbl.schema)

        return _from_tbl(tbl)

    def close(self):
        self.client.disconnect()
