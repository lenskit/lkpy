"""
Apache Arrow utilities.
"""

import numpy as np
import pyarrow as pa


def rb_to_structarray(rb: pa.RecordBatch):
    return pa.StructArray.from_arrays(rb.columns, rb.schema.names)


def tbl_to_structarray(tbl: pa.Table) -> pa.ChunkedArray | pa.StructArray:
    """
    Convert a PyArrow table to a struct array.
    """

    if hasattr(tbl, "to_struct_array"):
        return tbl.to_struct_array()
    else:
        batches = tbl.to_batches()
        if len(batches) == 1:
            return rb_to_structarray(batches[0])
        else:
            return pa.chunked_array([rb_to_structarray(rb) for rb in batches])  # type: ignore


def array_is_null(array: object) -> bool:
    if array is None:
        return True
    if isinstance(array, pa.Array) and not np.any(array.is_valid()):
        return True

    return False
