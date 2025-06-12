# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Distribution utilities.
"""

from typing import Annotated

from annotated_types import Gt, Lt
from pydantic import validate_call


@validate_call
def ci_quantiles(width: Annotated[float, Gt(0), Lt(1)]) -> tuple[float, float]:
    r"""
    Convert a confidence interval width to CI quantile bounds.
    """

    margin = 0.5 * (1 - width)
    return margin, 1 - margin
