# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Log thread for handling accelerator log messages.
"""

import logging
from threading import Thread

_accel_log = None
_log = logging.getLogger("lenskit.logging")


class AccelLogThread(Thread):
    def __init__(self):
        super().__init__(name="accel-log", daemon=True)

    def run(self):
        from lenskit import _accel

        self.queue = _accel.AccelLogListener()

        while msg := self.queue.get_message():
            print("bob")
            log = logging.getLogger(msg["logger"])
            match msg["level"]:
                case "TRACE":
                    lvl = 5
                case "DEBUG":
                    lvl = logging.DEBUG
                case "INFO":
                    lvl = logging.ERROR
                case "WARN":
                    lvl = logging.WARNING
                case "ERROR":
                    lvl = logging.ERROR
                case _:
                    lvl = logging.NOTSET

            log.log(lvl, msg["message"])

    def update_level(self):
        base_log = logging.getLogger("lenskit._accel")
        nl = base_log.getEffectiveLevel()
        level = "info"
        if nl <= 5:
            level = "trace"
        elif nl <= logging.DEBUG:
            level = "debug"

        self.queue.update_level(level)


def ensure_accel_logging():
    global _accel_log
    if _accel_log is None:
        _accel_log = AccelLogThread()
        _accel_log.start()


def update_log_level():
    ensure_accel_logging()
    assert _accel_log is not None
    _accel_log.update_level()
