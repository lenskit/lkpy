# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from time import perf_counter


class RateLimit:
    """
    Class for rate-limiting performance updates.

    Args:
        hertz:
            The update frequency in Hertz.
    """

    hertz: float
    _interval: float

    _last_update: float = 0.0

    def __init__(self, hertz: float = 5):
        self.hertz = hertz
        self._interval = 1 / hertz

    def want_update(self, now: float | None = None) -> bool:
        """
        Check if we are ready for an update.
        """
        if now is None:
            now = perf_counter()

        if now - self._last_update >= self._interval:
            return True
        else:
            return False

    def mark_update(self, now: float | None = None):
        """
        Mark that the system has been updated.
        """
        if now is None:
            now = perf_counter()
        self._last_update = now
