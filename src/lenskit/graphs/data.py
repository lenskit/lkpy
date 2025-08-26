# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Convert LensKit datasets to PyG graphs.
"""

from torch_geometric.data import HeteroData

from lenskit.data import Dataset


def torch_graph(ds: Dataset) -> HeteroData:
    pass
