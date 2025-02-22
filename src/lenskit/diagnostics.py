# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Warning and error classes and general LensKit diagnostic code.
"""


class DataWarning(UserWarning):
    """
    Warning raised for detectable problems with input data.
    """

    pass


class DataError(Exception):
    """
    Error raised for detectable problesms with input data.
    """

    pass


class FieldError(KeyError):
    """
    The requested field does not exist.
    """

    def __init__(self, entity, field):
        super().__init__(f"{entity}[{field}]")


class ConfigWarning(UserWarning):
    """
    Warning raised for detectable problems with component configurations.
    """

    pass


class PipelineError(Exception):
    """
    Pipeline configuration errors.

    .. note::

        This exception is only to note problems with the pipeline configuration
        and structure (e.g. circular dependencies).  Errors *running* the
        pipeline are raised as-is.
    """


class PipelineWarning(Warning):
    """
    Pipeline configuration and setup warnings.  We also emit warnings to the
    logger in many cases, but this allows critical ones to be visible even if
    the client code has not enabled logging.

    .. note::

        This warning is only to note problems with the pipeline configuration
        and structure (e.g. circular dependencies).  Errors *running* the
        pipeline are raised as-is.
    """
