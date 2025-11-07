# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# verify that the training data is correct
import pickle
import sys

from xopen import xopen

from lenskit.als import BiasedMFScorer
from lenskit.pipeline.nodes import ComponentInstanceNode

out_file = sys.argv[1]

with xopen(out_file, "rb") as pf:
    pipe = pickle.load(pf)

node = pipe.node("scorer")
assert isinstance(node, ComponentInstanceNode)
assert isinstance(node.component, BiasedMFScorer)
