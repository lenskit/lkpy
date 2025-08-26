# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Convert LensKit datasets to PyG graphs.
"""

import torch
from torch_geometric.data import HeteroData

from lenskit.data import Dataset
from lenskit.logging import get_logger

_log = get_logger(__name__)


def torch_graph(ds: Dataset) -> HeteroData:
    """
    Convert a LensKit dataset to a PyG heterogeneous graph.

    Stability:
        Experimental
    """
    log = _log.bind(dataset=ds.name)
    data = HeteroData()

    for entity in ds.schema.entities:
        log.debug("adding entities", type=entity)
        data[entity].n_id = ds.entities(entity).numbers()

    for rel in ds.schema.relationships:
        log.debug("adding relationships", type=rel)
        rset = ds.relationships(rel)
        entities = rset.entities
        if len(entities) != 2:
            log.warn("ignoring non-binary relationship %s", rel)
            continue

        src, tgt = entities
        tbl = rset.arrow(attributes=[])
        print(tbl)

        cols = [torch.tensor(c.to_numpy()) for c in tbl.columns]
        edges = torch.stack(cols)
        data[src, rel, tgt].edge_index = edges

    return data
