# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from typing import override

import numpy as np
from scipy.special import logsumexp

from lenskit.data import ItemList
from lenskit.data.accum import ValueStatAccumulator

from ._base import RankingMetricBase
from ._weighting import GeometricRankWeight


def log_rank_biased_precision(
    good: np.ndarray, log_weights: np.ndarray, log_normalization: float
) -> float:
    r"""
    Compute the natural logarithm of rank-biased precision from log-space weights.

    This is the log-space analogue of
    :func:`~lenskit.metrics.ranking.rank_biased_precision`:

    .. math::
        \ln \operatorname{RBP} = \operatorname{logsumexp}_{i \in \text{good}}
        \ln w_i - \ln Z

    Working in log space means the rank discounts are never exponentiated and then
    summed, so scores far below the ``float64`` subnormal floor remain representable
    instead of collapsing to zero.  A list that retrieves none of its test items has
    :math:`\operatorname{RBP} = 0`, and therefore returns :math:`-\infty` — a valid
    measurement, not a missing one.

    Args:
        good:
            Boolean array indicating relevant items at each position.
        log_weights:
            Natural log of the weight for each item position (same length as
            ``good``).
        log_normalization:
            Natural log of the normalization factor.

    Returns:
        The natural log of the RBP score, or ``-inf`` when no position is relevant.

    Stability:
        Experimental
    """
    if not np.any(good):
        return -np.inf

    return float(logsumexp(log_weights[good]) - log_normalization)


class LogRBP(RankingMetricBase):
    r"""
    Evaluate recommendations with the natural logarithm of rank-biased precision,
    aggregated as :math:`\log(\overline{\operatorname{RBP}})`.

    This metric measures the same quantity as :class:`~lenskit.metrics.RBP`, but
    carries each list's score in log space so that the summary can be computed as
    :math:`\log(\operatorname{mean}(\operatorname{RBP}))` rather than
    :math:`\operatorname{mean}(\operatorname{RBP})`.  The two are different
    quantities: the summary is obtained by log-sum-exp accumulation,

    .. math::
        \log\!\left(\frac{1}{n}\sum_i \operatorname{RBP}_i\right)
        = \operatorname{logsumexp}_i \ln \operatorname{RBP}_i - \ln n,

    which keeps resolving for scores far below the ``float64`` subnormal floor.  With
    geometric weighting and patience :math:`p`, a single relevant item at rank
    :math:`r` scores :math:`(1-p)p^{r-1}`; at :math:`p=0.85` and :math:`r=5001` this
    underflows to exactly ``0.0`` in normal space (its log is :math:`\approx -814.5`)
    and is therefore still finite here.

    Only the geometric rank weighting used by RBP is supported, with the same
    normalization semantics.  Custom :class:`~lenskit.metrics.RankWeight` objects and
    ``weight_field`` are deliberately **not** accepted: the base
    :meth:`~lenskit.metrics.RankWeight.log_weight` falls back to
    :math:`\ln w = \ln(e^{\ln w})`, which exponentiates the very discounts this metric
    exists to avoid.

    Like :class:`~lenskit.metrics.RBP`, it distinguishes two different cases that must
    not be conflated:

    * a list with test items but no retrieved items scores :math:`0`, and is reported
      as :math:`-\infty` and **retained** in the aggregate (it lowers the mean);
    * a list with an **empty test list** has no defined score, is reported as
      ``NaN``, and is **excluded** from both the numerator and the denominator —
      matching :class:`~lenskit.metrics.RBP` and the existing accumulator behaviour.

    Consequently a collection whose valid lists all score zero aggregates to
    :math:`-\infty`, while a collection with no valid lists at all aggregates to
    ``NaN`` rather than a fabricated zero.

    .. note::

        The summary key is provisionally named ``logmean`` and appears in collected
        results as ``LogRBP.logmean``.  The name is not settled upstream; see
        :issue:`1159`.

    .. note::

        **Degenerate case, deliberate divergence from** :class:`~lenskit.metrics.RBP`.
        With ``normalize=True`` and an *empty recommendation list*, the normalizing
        sum is empty, so :class:`~lenskit.metrics.RBP` divides by zero and raises
        :exc:`ZeroDivisionError`.  This metric instead returns ``NaN``, treating the
        score as undefined rather than raising.  Every other path matches
        :class:`~lenskit.metrics.RBP` exactly: with ``normalize=False`` an empty list
        scores :math:`0` and is therefore reported here as :math:`-\infty` and
        retained, just as :class:`~lenskit.metrics.RBP` reports ``0.0``.  Both
        behaviors are pinned by tests
        (``test_logrbp_empty_recs_normalize_returns_nan`` asserts each side).

    .. warning::

        Experimental prototype.  The ``normalize`` option inherits RBP's
        experimental normalization.

    Args:
        n:
            The maximum recommendation list length.
        patience:
            The patience parameter :math:`p`, the probability that the user
            continues browsing at each point.  The default is 0.85.
        normalize:
            Whether to normalize by the maximum achievable with the test data, as
            :class:`~lenskit.metrics.RBP` does.
        k:
            Deprecated alias for ``n``.

    Stability:
        Experimental
    """

    weight: GeometricRankWeight
    patience: float
    normalize: bool

    def __init__(
        self,
        n: int | None = None,
        *,
        k: int | None = None,
        patience: float = 0.85,
        normalize: bool = False,
    ):
        super().__init__(n, k=k)
        self.patience = patience
        self.weight = GeometricRankWeight(patience)
        self.normalize = normalize

    @override
    def measure_list(self, recs: ItemList, test: ItemList) -> float:
        recs = self.truncate(recs)
        k = len(recs)

        nrel = len(test)
        if nrel == 0:
            return np.nan

        good = recs.isin(test)

        ranks = recs.ranks()
        assert ranks is not None

        # the discounts are never exponentiated: log weights are produced directly
        log_weights = self.weight.log_weight(ranks)

        wmax = self.weight.series_sum()
        assert wmax is not None
        if self.normalize:
            # normalize by the log of the maximum achievable RBP
            max_slice = log_weights[: min(nrel, k)]
            if max_slice.size == 0:
                # the normalizing sum is empty; RBP divides by zero here and raises
                # ZeroDivisionError.  We report an undefined measurement instead.
                return np.nan
            log_normalization = float(logsumexp(max_slice))
        else:
            # normalization defined by the metric: the infinite series sum
            log_normalization = math.log(wmax)

        return log_rank_biased_precision(good, log_weights, log_normalization)

    @override
    def extract_list_metrics(self, data: float) -> float:
        """
        Expose the per-list log-RBP value in the collected list metrics.
        """
        return data

    @override
    def create_accumulator(self) -> LogRBPAccumulator:
        r"""
        Create the accumulator implementing this metric's summary semantics.

        The aggregate is :math:`\log(\operatorname{mean}(\operatorname{RBP}))`, computed
        as :math:`\operatorname{logsumexp}_i \ell_i - \ln n_{\text{valid}}` over the
        retained per-list log measurements :math:`\ell_i`.  A measurement is retained
        when it is neither ``None`` nor ``NaN``; :math:`-\infty` **is retained**, since
        it encodes a genuine zero-valued RBP and contributes :math:`e^{-\infty} = 0`
        to the sum while still counting in the denominator.  That is what separates
        this aggregate from the mean of the logs, which would be :math:`-\infty`
        whenever any list scores zero.

        Returns ``n`` and ``logmean`` only.  It deliberately reports no ``mean``,
        ``median`` or ``std`` of the log values: those statistics are not meaningful
        once :math:`-\infty` is a legitimate observation.  ``logmean`` is ``NaN`` when
        there are no valid measurements, and :math:`-\infty` when every valid
        measurement came from a zero-scoring list.

        Storage is delegated to :class:`~lenskit.data.accum.ValueStatAccumulator`,
        whose ``add`` drops exactly ``None`` and ``NaN`` and keeps every other value.
        ``test_accumulator_drops_missing_keeps_neginf`` pins that :math:`-\infty` is
        retained, so a future narrowing of ``add`` to finite values fails loudly
        instead of silently discarding zero-RBP lists.
        """
        return LogRBPAccumulator()


class LogRBPAccumulator:
    # Deliberately undocumented at class and method level: with
    # autoapi_own_page_level = "class" plus undoc-members, a documented class here gets
    # its own API page that no toctree includes (a toc.not_included warning), and
    # upstream's other private accumulators (GiniAccumulator, UniqueItemAccumulator,
    # AvgErrorAccumulator) are undocumented for the same reason.  The semantics are
    # documented on LogRBP.create_accumulator, where callers actually read them.

    _values: ValueStatAccumulator

    def __init__(self) -> None:
        self._values = ValueStatAccumulator()

    @property
    def values(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self._values.values

    def __len__(self) -> int:
        return len(self._values)

    def add(self, value: float | None) -> None:
        # None and NaN are excluded, -inf is retained: that is exactly
        # ValueStatAccumulator.add's rule, so storage and the retain rule are not
        # duplicated here.  Pinned by test_accumulator_drops_missing_keeps_neginf,
        # which fails loudly if upstream ever narrows add() to finite values only.
        self._values.add(value)

    def accumulate(self) -> dict[str, float]:
        # logsumexp over retained log scores minus ln n  ==  log(mean(RBP)).
        # No valid measurements -> NaN (undefined), never a fabricated zero.
        n = len(self)
        if n == 0:
            return {"n": 0, "logmean": np.nan}

        return {"n": n, "logmean": float(logsumexp(self.values) - math.log(n))}
