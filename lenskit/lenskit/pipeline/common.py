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
