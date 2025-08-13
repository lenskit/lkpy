# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Distribution utilities.
"""

from typing import Annotated

import numpy as np
from annotated_types import Gt, Lt
from pydantic import validate_call
from scipy import stats


@validate_call
def ci_quantiles(
    width: Annotated[float, Gt(0), Lt(1)], *, expand: Annotated[int, Gt(1)] | None = None
) -> tuple[float, float]:
    r"""
    Convert a confidence interval width to CI quantile bounds.

    Args:
        width:
            The CI interval width.
        expand:
            If not ``None``, a sample size :math:`n` to use to
            expand the CI as in the expanded percentile bootstrap.
    """

    margin = 0.5 * (1 - width)
    if expand:
        factor = np.sqrt(expand / (expand - 1))
        # get t_(alpha/2),n-1
        t = stats.t.ppf(margin, expand - 1)
        # get standard normal CDF
        margin = stats.norm.cdf(factor * t)

    return margin, 1 - margin
