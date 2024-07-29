# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import pandas as pd

from . import Pipeline
from .components import Component


def topn_pipeline(scorer: Component[pd.Series]) -> Pipeline:
    """
    Create a pipeline that produces top-N recommendations using the specified
    scorer.  The scorer should have the following call signature::

        def scorer(user: UserHistory, items: ItemList) -> pd.Series: ...
    """
    raise NotImplementedError()
