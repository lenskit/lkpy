from pydantic import BaseModel

from lenskit.data import ItemList
from lenskit.pipeline import Component


class LinearBlendConfig(BaseModel):
    "Configuration for :class:`LinearBlendScorer`."

    # define the parameter with a type, default value, and docstring.
    mix_weight: float = 0.5
    r"""
    Linear blending mixture weight :math:`\alpha`.
    """


class LinearBlendScorer(Component[ItemList]):
    r"""
    Score items with a linear blend of two other scores.

    Given a mixture weight :math:`\alpha` and two scores
    :math:`s_i^{\mathrm{left}}` and :math:`s_i^{\mathrm{right}}`, this
    computes :math:`s_i = \alpha s_i^{\mathrm{left}} + (1 - \alpha)
    s_i^{\mathrm{right}}`.  Missing values propagate, so only items
    scored in both inputs have scores in the output.
    """

    # define the configuration attribute, with a docstring to make sure
    # it shows up in component docs.
    config: LinearBlendConfig
    "Configuration parameters for the linear blend."

    # the __call__ method defines the component's operation
    def __call__(self, left: ItemList, right: ItemList) -> ItemList:
        """
        Blend the scores of two item lists.
        """
        ls = left.scores("pandas", index="ids")
        rs = right.scores("pandas", index="ids")
        ls, rs = ls.align(rs)
        alpha = self.config.mix_weight
        combined = ls * alpha + rs * (1 - alpha)
        return ItemList(item_ids=combined.index, scores=combined.values)
