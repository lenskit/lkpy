# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from torch_geometric.data import HeteroData

from lenskit.data import Dataset
from lenskit.graphs.data import torch_graph


def test_ml_graph(ml_ds: Dataset):
    g = torch_graph(ml_ds)

    assert g["user"].num_nodes == ml_ds.user_count
    assert g["item"].num_nodes == ml_ds.item_count
    assert g.num_edges == ml_ds.interaction_count
