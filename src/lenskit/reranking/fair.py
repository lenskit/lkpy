# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
FA*IR top-N re-ranking.
"""

# pyright: basic
from __future__ import annotations

from collections import deque

import numpy as np
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt
from scipy.stats import binom
from typing_extensions import override

from lenskit.data import Dataset, ItemList
from lenskit.logging import Stopwatch, get_logger
from lenskit.pipeline import Component
from lenskit.training import Trainable, TrainingOptions

_log = get_logger(__name__)


class FAIRRerankerConfig(BaseModel, extra="forbid"):
    """
    Configuration for :class:`FAIRReranker`.
    """

    n: PositiveInt
    """
    Size of the top-N subset to rerank. If ``None``, the entire
    candidate list is reranked.
    """

    p: PositiveFloat = Field(0.5, gt=0.0, lt=1.0)
    """
    The target proportion of protected items in the list (0.0 ≤ p ≤ 1.0).
    The default value is 0.5.
    """

    alpha: PositiveFloat = Field(0.1, gt=0.0, lt=1.0)
    """
    The probability threshold for Type I error across prefixes.
    The default value is 0.1.
    """

    protected_attribute: str = Field(
        "protected", description="Item attribute name for the protected flag"
    )
    """
    The item attribute name that marks whether an item is protected.
    The default name is 'protected'.
    """


class FAIRReranker(Component[ItemList], Trainable):
    """
    FA*IR top-N re-ranking algorithm.

    Re-ranks a sorted candidate list to satisfy binomial lower-bound constraints
    for a binary protected/unprotected attribute at every prefix (1..N). It expects
    a sorted list of items, the target proportion of protected items ``p`` and
    significance level ``alpha``.

    For details, see `Zehlike et al. (2017)`_.

    .. _Zehlike et al. (2017): https://doi.org/10.1145/3132847.3132938

    Stability:
        Caller

    """

    config: FAIRRerankerConfig
    "FA*IR reranker configuration."

    def _compute_m_list(self, n, p, alpha):
        n_vals = np.arange(1, n + 1)
        # if we see m or fewer protected, it’s rare enough (probability ≤ α) that we call it unfair.
        m_list_ = binom.ppf(alpha, n_vals, p)
        # set bounds for edge cases (<0 or >N)
        m_list_ = np.clip(m_list_, 0, n_vals).astype(int)
        return m_list_

    def _compute_blocks(self, m_list_):
        max_m = int(m_list_[-1])
        if max_m == 0:
            return []

        # find the positions where m increases to form blocks
        # (last value of each block increases by 1)
        change_points = np.flatnonzero(np.diff(m_list_, prepend=0)) + 1
        block_sizes = np.diff(change_points, prepend=0)
        return block_sizes

    def _compute_rejection_prob(self, n, p, alpha_c):
        """
        Algorithm 1 (AdjustSignificance), exactly:
        - compute m(pos) from Binomial PPF at alpha_c
        - compute blocks b(j)
        - for j=1..m(n):
            S = conv(S, Bin(b(j), p))
            S[j-1] = 0     # prune bucket j-1
        Returns rejection probability = 1- sum(S).
        """
        m_list_ = self._compute_m_list(n, p, alpha_c)
        m_blocks = self._compute_blocks(m_list_)
        S = np.array([1.0], dtype=np.float64)

        if len(m_blocks) > 0:
            for j, block_size in enumerate(m_blocks, start=1):
                if block_size not in self.pmf_cache:
                    self.pmf_cache[block_size] = binom.pmf(np.arange(block_size + 1), block_size, p)

                S = np.convolve(self.pmf_cache[block_size], S)
                # set the next index (j-1) to zero since the last element of each block
                # increases by 1 and now we need at least j protected up to this point
                S[j - 1] = 0

        return float(1 - S.sum())

    def _binary_search_significance(self, n, p, alpha, tolerance=1e-10, max_iter=100):
        low, high = 0, 1

        for _ in range(max_iter):
            alpha_c = (low + high) / 2
            rejection_prob = self._compute_rejection_prob(n, p, alpha_c)

            if rejection_prob > alpha:
                high = alpha_c
            else:
                low = alpha_c
            if abs(rejection_prob - alpha) < tolerance or high - low < tolerance:
                break

        alpha_c = (low + high) / 2
        return alpha_c

    @override
    def train(self, data: Dataset, options: TrainingOptions = TrainingOptions()):
        """
        Precompute adjusted alpha and m-table for the configured (n,p,alpha)

        Training computes the (multiple-tests–corrected) prefix requirements m[i]
        so that every prefix of size i has at least m[i] protected items with
        significance α (after correction).

        Args:
            protected attributes:
                The item entity table must have a scalar attribute ``protected`` that is ``True``
                if the item is in the protected group and ``False`` otherwise.
        """
        self.pmf_cache = {}

        log = _log.bind(alpha=self.config.alpha, n_max=self.config.n)
        timer = Stopwatch()
        log.info("computing FA*IR thresholds")

        # 1. find α_c
        self.alpha_c_ = self._binary_search_significance(
            n=self.config.n, p=self.config.p, alpha=self.config.alpha
        )
        # 2. compute m_table
        self.m_list_ = self._compute_m_list(n=self.config.n, p=self.config.p, alpha=self.alpha_c_)

        items = data.entities("item")
        protected = self.config.protected_attribute

        if protected not in items.attributes:
            raise ValueError(
                f"Dataset is missing required '{protected}' attribute for item entities"
            )

        prot = items.attribute(protected).numpy()
        self.protected_attributes = np.equal(prot, True)
        self.vocab = items.vocabulary

        log.info(
            "[%s] computed thresholds with p=%.2f, alpha=%.2f → alpha_c=%.8f",
            timer,
            self.config.p,
            self.config.alpha,
            self.alpha_c_,
        )

    def __call__(self, items: ItemList, n: int | None = None) -> ItemList:
        list_size = len(items)
        item_numbers = items.numbers(vocabulary=self.vocab, missing="negative")

        is_protected = np.full(list_size, False)

        mask_vocab_items = item_numbers >= 0
        is_protected[mask_vocab_items] = self.protected_attributes[item_numbers[mask_vocab_items]]

        p_items = deque(np.nonzero(is_protected)[0])
        up_items = deque(np.nonzero(np.logical_not(is_protected))[0])

        count_prot = 0
        reranked_indices = []
        n_config = self.config.n

        # check that runtime n = configured n
        if n is not None:
            if n > n_config:
                raise ValueError(
                    f"The requested rerank length n={n}, exceeds configured n={n_config}."
                )
            elif n < n_config:
                _log.warning(
                    "FA*IR: model trained for n=%d used on n=%d; fairness test might be violated.",
                    n_config,
                    n,
                )

        # use provided n or fallback to config.n (never larger than list size)
        n = min(n or n_config, list_size)

        # rerank the list
        for i in range(n):
            if count_prot < self.m_list_[i] and p_items:
                reranked_indices.append(p_items.popleft())
                count_prot += 1
            elif p_items and up_items:
                if p_items[0] < up_items[0]:
                    reranked_indices.append(p_items.popleft())
                    count_prot += 1
                else:
                    reranked_indices.append(up_items.popleft())
            elif up_items:
                reranked_indices.append(up_items.popleft())
            else:
                reranked_indices.append(p_items.popleft())
                count_prot += 1

        topn_reranked = ItemList(items[reranked_indices], ordered=True)
        return topn_reranked
