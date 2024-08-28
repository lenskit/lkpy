# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT


from . import Pipeline
from .components import Component


def topn_pipeline(scorer: Component, *, predicts_ratings: bool = False) -> Pipeline:
    """
    Create a pipeline that produces top-N recommendations using the specified
    scorer.  The scorer should have the following call signature::

        def scorer(user: UserHistory, items: ItemList) -> pd.Series: ...

    Args:
        scorer:
            The scorer to use in the pipeline (it will added with the component
            name ``score``, see :ref:`pipeline-names`).
        predicts_ratings:
            If ``True``, make ``predict-ratings`` an alias for ``score`` so that
            evaluation components know this pipeline can predict ratings.
    """
    raise NotImplementedError()
