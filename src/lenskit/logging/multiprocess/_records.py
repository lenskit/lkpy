# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Support for arbitrary record sinks.
"""

from typing import Generic, Protocol, TypeVar
from uuid import UUID

R = TypeVar("R", contravariant=True)


class RecordSink(Protocol, Generic[R]):
    """
    Generic interface for record sinks.

    Stability:
        Internal
    """

    sink_id: UUID

    def record(self, record: R, /):
        """
        Record the timings for a pipeline run.
        """
