"""
Support for sharing objects (e.g. trained models) between processes.
"""

import sys
import os.path
from abc import ABCMeta, abstractmethod
from pathlib import Path
import logging
import uuid
import json
import tempfile
import subprocess
import pickle

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
    Base class for shared object repositories.
    """

    def __init__(self, cache_limit=32):
        self._cache_limit = 32
        self._cache = {}

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
        raise NotImplementedError()

    def resolve(self, key):
        """
        Resolve a key to an object from the repository.

        Args:
            key: the object key to resolve or retrieve.

        Returns:
            a reference to or copy of the shared object.  Client code must not try to modify
            this object.
        """
        result = self._cache.get(key)

        if result is None:
            result = self.read_object(key)
            self._cache[key] = result
            self._clean_cache()

        return result

    @abstractmethod
    def client(self):
        """
        Return a client copy of this repository.  Client repository connections are intended
        to be consumers of the objects shared in the original repository, and they can be
        pickled and cloned across processes.  They are not guaranteed to have access to data
        once the original repository object is closed.

        Returns:
            ObjectRepo: a client copy of this repository.  If this repository is already usable
            as a client, this method may return ``self``.
        """
        raise NotImplementedError()

    @abstractmethod
    def read_object(self, key):
        """
        Resolve a key to an object from the repository.

        Args:
            key: the object key to resolve or retrieve.

        Returns:
            a reference to or copy of the shared object.  Client code must not try to modify
            this object.
        """
        raise NotImplementedError()

    def clear(self):
        """
        Clear objects shared to this repo (since the last :py:meth:`clear` call).  Useful for
        freeing up memory. This is an optional operation that may be a no-op.  It will only
        clear objects saved in *this* repository object; deleted objects will also be deleted
        for clients & clones, but objects shared via a client or clone will be unaffected.
        """
        pass

    def close(self):
        """
        Close down the repository.
        """
        self._cache.clear()

    def _clean_cache(self):
        if len(self._cache) > self._cache_limit:
            del_keys = []
            for k in self._cache.keys():
                # 2 references: one in the cache, one here
                if sys.getrefcount(self._cache[k]) < 3:
                    del_keys.append(k)

            _logger.debug('evicting %d keys', del_keys)
            for k in del_keys:
                del self._cache[k]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getstate__(self):
        """
        Return picklable state.  This method removes the ``_cache``; subclasses should override,
        call this, and do additional cleanup on its results.
        """
        state = self.__dict__.copy()
        del state['_cache']
        return state

    def __setstate__(self, state):
        """
        Initialize from a picklable state. This implementation creates a fresh cache.
        """
        self.__dict__.update(state)
        self._cache = {}


class FileRepo(ObjectRepo):
    """
    Share objects by serializing them to Arrow files in a directory.  Keys are UUIDs.
    """

    def __init__(self, dir=None):
        super().__init__()
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
        super().close()

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

        _logger.debug('saved %s (%d bytes)', key, fn.stat().st_size)

        return key

    def read_object(self, key):
        fn = self.dir / '{}.parquet'.format(key)
        if not fn.exists():
            raise KeyError(key)

        _logger.debug('reading %s from %s (%d bytes)', repr(object), key,
                      fn.stat().st_size)

        pqf = parq.ParquetFile(str(fn))

        tbl = pqf.read()

        return _from_tbl(tbl)

    def client(self):
        if self._tmpdir is None:
            # no resources, we are a client
            return self
        else:
            return FileRepo(self.dir)

    def __getstate__(self):
        if self._tmpdir is not None:
            raise pickle.PicklingError("non-client repos cannot be pickled")
        return super().__getstate__()

    def __str__(self):
        dstate = ' (self-deleting)' if self._tmpdir is not None else ''
        return '<FileRepo {}{}>'.format(self.dir, dstate)


class PlasmaRepo(ObjectRepo):
    """
    Share objects by serializing them to Arrow files in a directory.  Keys are UUIDs.
    """

    _proc = None
    _dir = None
    _shared = None

    def __init__(self, socket=None, manager="", release_delay=0, size=None):
        super().__init__()
        if socket is None and size is None:
            raise ValueError("must specify one of size and socket")

        if socket is None:
            self._dir = tempfile.TemporaryDirectory(prefix='lk-repo')
            socket = os.path.join(self._dir.name, 'plasma.sock')
            self._proc = subprocess.Popen(['plasma_store', '-m', str(size), '-s', socket])

        self._socket = socket
        self._manager = manager
        self._release_delay = release_delay
        self._plasma_client = plasma.connect(socket, manager, release_delay)
        self._shared = []

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

        buf = self._plasma_client.create(id, size)
        stream = pa.FixedSizeBufferWriter(buf)
        sw = pa.RecordBatchStreamWriter(stream, schema)
        sw.write_table(tbl)
        sw.close()
        self._plasma_client.seal(id)

        _logger.debug('shared object to %s', id)
        self._shared.append(id)
        return id

    def read_object(self, key):
        _logger.debug('resolving object %s', key)
        [data] = self._plasma_client.get_buffers([key])
        buf = pa.BufferReader(data)
        _logger.debug('reading object of size %d', buf.size())
        reader = pa.RecordBatchStreamReader(buf)
        tbl = reader.read_all()
        _logger.debug('loaded table with schema %s', tbl.schema)

        return _from_tbl(tbl)

    def close(self):
        super().close()
        if self._plasma_client is not None:
            self._plasma_client.disconnect()
            self._plasma_client = None

        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _logger.warn('plasma client %d did not respond to SIGTERM, killing', self._proc.pid)
                self._proc.kill()

            self._proc = None

    def clear(self):
        _logger.info('clearing repository %s', self)
        while self._shared:
            key = self._shared.pop()
            _logger.debug('releasing object %s', key)
            self._plasma_client.release(key)

    def client(self):
        return PlasmaRepo(self._socket, self._manager, self._release_delay)

    def __getstate__(self):
        if self._proc is not None:
            raise pickle.PicklingError('process-owning stores cannot be pickled')
        state = super().__getstate__()
        del state['_plasma_client']
        del state['_shared']
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._plasma_client = plasma.connect(self._socket, self._manager, self._release_delay)
        self._shared = []

    def __str__(self):
        if self._proc is None:
            cxn = self._socket
        else:
            cxn = 'pid {}'.format(self._proc.pid)
        return '<PlasmaRepo {}>'.format(cxn)

    def __del__(self):
        self.close()


def repo(capacity):
    """
    Create the best available sharing repository for this platform.

    Args:
        capacity: the required capacity in bytes.

    Returns:
        ObjectRepo: a repository.
    """

    if sys.platform == 'win32':
        return FileRepo()
    else:
        return PlasmaRepo(size=capacity)
