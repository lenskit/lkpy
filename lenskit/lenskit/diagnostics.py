# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

class DataWarning(UserWarning):
    """
    Warning raised for detectable problems with input data.
    """

    pass


class ConfigWarning(UserWarning):
    """
    Warning raised for detectable problems with algorithm configurations.
    """

    pass
