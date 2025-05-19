# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

# pyright: strict
from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from ..multiprocess._protocol import ProgressMessage


class Progress:
    """
    Base class for progress reporting.  The default implementations do nothing.
    """

    uuid: UUID
    total: int | float | None

    def __init__(self, *args: Any, uuid: UUID | None = None, **kwargs: Any):
        self.uuid = uuid if uuid is not None else uuid4()

    @classmethod
    def handle_message(cls, update: ProgressMessage):
        pass

    def update(
        self,
        advance: int = 1,
        completed: int | None = None,
        total: int | None = None,
        **kwargs: float | int | str,
    ):
        """
        Update the progress bar.

        Args:
            advance:
                The amount to advance by.
            completed:
                The number completed; this overrides ``advance``.
            total:
                A new total, to update the progress bar total.
        """
        pass

    def finish(self):
        """
        Finish and clean up this progress bar.  If the progresss bar is used as
        a context manager, this is automatically called on context exit.
        """
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args: Any):
        self.finish()
