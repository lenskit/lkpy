# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Observation propensity models for IPS-weighted evaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, override

import numpy as np
import pandas as pd

from lenskit.data import Dataset, ItemList, ItemListCollection
from lenskit.data.types import NPVector


class PropensityModel(ABC):
    r"""
    Base class for observation propensity models.

    A propensity model maps items to the probability :math:`P_i` that a user's
    interaction with the item is observed, and to the inverse propensity weight
    :math:`w_i = 1 / P_i` applied by IPS estimators.

    Propensities are identified only up to a constant of proportionality
    :cite:p:`yangUnbiasedOfflineRecommender2018`.  Self-normalized estimators
    are invariant to that constant; unnormalized ones are not, so subclasses
    should document the scale they adopt.

    Stability:
        Caller
    """

    @abstractmethod
    def propensities(self, items: ItemList) -> NPVector[np.float64]:
        """
        Compute the observation propensity of each item, in :math:`(0, 1]`.

        This is normally applied to the test items, since propensities describe
        the observation of relevant items.
        """

    def weights(self, items: ItemList) -> NPVector[np.float64]:
        """
        Compute the inverse propensity weight of each item.
        """
        return np.reciprocal(self.propensities(items))


class UniformPropensity(PropensityModel):
    r"""
    Treat every item as equally likely to be observed (:math:`P_i = 1`).

    This is the missing-at-random assumption that
    :cite:t:`yangUnbiasedOfflineRecommender2018` show does not hold in practice.
    An IPS metric using it reduces to its uncorrected counterpart, making it the
    reference point for measuring how much the correction moves a score.

    Stability:
        Caller
    """

    @override
    def propensities(self, items: ItemList) -> NPVector[np.float64]:
        return np.ones(len(items), dtype=np.float64)


class PopularityPropensity(PropensityModel):
    r"""
    Estimate observation propensities from item popularity, following the
    select-then-interact model of :cite:t:`yangUnbiasedOfflineRecommender2018`.

    An item is observed if a recommender presents it and the user then interacts
    with it.  Modelling presentation as a power law in observed popularity and
    eliminating the unobservable true popularity gives

    .. math::
        \hat{P}_i \propto (n^*_i)^{\frac{\gamma + 1}{2}}

    for observed interaction count :math:`n^*_i`.  We take the constant of
    proportionality that puts the most popular item at :math:`\hat{P}_i = 1`:

    .. math::
        \hat{P}_i = \left(\frac{n^*_i}{n^*_{\max}}\right)^{\frac{\gamma+1}{2}}

    so that every weight is at least 1.  Items with no interactions, and items
    absent from ``data`` entirely, are counted as one interaction and therefore
    take the largest weight.

    ``gamma`` is the exponent as defined in the paper; the :math:`(\gamma+1)/2`
    correction is applied here.  It has no default because it depends on both
    the dataset and the algorithm that produced the log: reported values run
    from about 1.5 to 3, and :func:`estimate_power_law_gamma` will fit it.
    :math:`\gamma = -1` gives uniform propensities, but prefer
    :class:`UniformPropensity` for that.

    Estimate propensities from training data.  The unbiasedness argument holds
    the propensities fixed with respect to the observations being weighted,
    which fails when they are drawn from the same test observations.

    Args:
        data:
            The dataset, normally the training split, supplying item popularity.
        gamma:
            The power-law exponent :math:`\gamma`.
        count:
            Whether to count distinct users or total interactions.
        max_weight:
            Cap on inverse propensity weights, for variance control.
        clip_quantile:
            Cap weights at this quantile of the weight distribution over the
            items of ``data``.  Applied together with ``max_weight``.

    Stability:
        Caller
    """

    gamma: float
    "The power-law exponent."
    exponent: float
    "The exponent applied to observed popularity, :math:`(\\gamma+1)/2`."
    item_propensities: pd.Series
    "Propensity of each item in the source dataset."
    missing_propensity: float
    "Propensity of items absent from the source dataset."

    def __init__(
        self,
        data: Dataset,
        *,
        gamma: float,
        count: Literal["users", "interactions"] = "users",
        max_weight: float | None = None,
        clip_quantile: float | None = None,
    ):
        if gamma < -1:
            raise ValueError("gamma must be at least -1")
        if max_weight is not None and max_weight < 1:
            raise ValueError("max_weight must be at least 1")
        if clip_quantile is not None and not 0 < clip_quantile < 1:
            raise ValueError("clip_quantile must be in (0, 1)")

        self.gamma = gamma
        self.exponent = (gamma + 1) / 2

        counts = _item_counts(data, count).clip(lower=1)
        top = counts.max()
        weights = (top / counts) ** self.exponent
        unseen = top**self.exponent

        cap = np.inf
        if clip_quantile is not None:
            cap = np.quantile(weights, clip_quantile)
        if max_weight is not None:
            cap = min(cap, max_weight)

        self.item_propensities = np.reciprocal(weights.clip(upper=cap))
        self.missing_propensity = 1 / min(unseen, cap)

    @override
    def propensities(self, items: ItemList) -> NPVector[np.float64]:
        if not len(items):
            return np.zeros(0)

        ps = self.item_propensities.reindex(items.ids(), fill_value=self.missing_propensity)
        return ps.fillna(self.missing_propensity).to_numpy(dtype=np.float64)


class FieldPropensity(PropensityModel):
    """
    Read observation propensities from a field on the test item list.

    This supports propensities obtained outside LensKit, such as from a logging
    policy or an exposure model, instead of from item popularity.

    Args:
        field:
            The name of the item list field holding the propensities.

    Stability:
        Caller
    """

    field: str

    def __init__(self, field: str = "propensity"):
        self.field = field

    @override
    def propensities(self, items: ItemList) -> NPVector[np.float64]:
        ps = items.field(self.field)
        if ps is None:
            raise KeyError(f"test items have no field {self.field}")
        if len(ps) and not np.all(np.isfinite(ps) & (ps > 0)):
            raise ValueError(f"field {self.field} has non-positive propensities")

        return np.asarray(ps, dtype=np.float64)


def estimate_power_law_gamma(
    recs: ItemListCollection,
    data: Dataset,
    *,
    n: int | None = 50,
    count: Literal["users", "interactions"] = "users",
    exclude_quantile: float = 0.005,
) -> float:
    r"""
    Fit the power-law exponent :math:`\gamma` of a recommender's presentation
    bias :cite:p:`yangUnbiasedOfflineRecommender2018`.

    Let :math:`f(n^*)` be the average number of times an item of observed
    popularity :math:`n^*` is recommended.  Presentation is modelled as
    :math:`\hat{P}^{\mathrm{select}}_i \propto (n^*_i)^\gamma \propto f(n^*_i)`,
    so :math:`\gamma` is the slope of :math:`\log f` against :math:`\log n^*`.
    The paper writes the fit as a sum over ordered pairs of popularity levels,
    which is the least-squares slope computed here.

    The result is the ``gamma`` accepted by :class:`PopularityPropensity`.  It
    is unreliable for recommenders that show nearly the same list to every user,
    since then :math:`f` is supported on only the most popular items.

    Args:
        recs:
            Recommendations from the algorithm being characterized.
        data:
            The dataset supplying observed item popularity.
        n:
            Truncate recommendation lists to this length; the paper uses 50.
        count:
            Whether to count distinct users or total interactions.
        exclude_quantile:
            Drop this fraction of the most popular items before fitting; the
            paper drops the top 0.5%.

    Returns:
        The estimated exponent :math:`\gamma`.

    Stability:
        Caller
    """
    if not 0 <= exclude_quantile < 1:
        raise ValueError("exclude_quantile must be in [0, 1)")

    counts = _item_counts(data, count)
    counts = counts[counts > 0]
    shown = [items[:n].ids() for _, items in recs if len(items)]
    if not shown:
        raise ValueError("no recommendations to fit")

    freqs = pd.Series(np.concatenate(shown)).value_counts()
    freqs = freqs.reindex(counts.index, fill_value=0).astype(np.float64)

    if exclude_quantile:
        keep = counts <= counts.quantile(1 - exclude_quantile)
        counts, freqs = counts[keep], freqs[keep]

    levels = freqs.groupby(counts).mean()
    levels = levels[levels > 0]
    if len(levels) < 2:
        raise ValueError("need at least two popularity levels with recommendations")

    x = np.log(levels.index.to_numpy(dtype=np.float64))
    y = np.log(levels.to_numpy(dtype=np.float64))
    x = x - x.mean()

    return (x @ (y - y.mean()) / (x @ x)).item()


def _item_counts(data: Dataset, count: Literal["users", "interactions"]) -> pd.Series:
    stats = data.item_stats()
    match count:
        case "users":
            counts = stats["user_count"]
        case "interactions":
            counts = stats["count"]
        case _:  # pragma: nocover
            raise ValueError(f"invalid count {count}")

    if not len(counts):
        raise ValueError("dataset has no items")

    return counts.astype(np.float64)
