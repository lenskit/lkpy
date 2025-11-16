# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
LensKit logging processors and converters.
"""

import multiprocessing as mp
import re
from collections import deque
from datetime import datetime
from typing import Any

import structlog
from structlog.typing import EventDict

LOGGED_ERRORS = deque([], 5)
_WARN_CODE_RE = re.compile(r"\s+\((LKW-\w+)\)$")


def filter_exceptions(logger: Any, method: str, event_dict: EventDict) -> EventDict:
    exc = event_dict.get("exc_info", None)
    if isinstance(exc, Exception):
        count = getattr(exc, "_lenskit_seen", 0)
        exc._lenskit_seen = count + 1
        if count > 1:
            del event_dict["exc_info"]

    return event_dict


def remove_internal(logger: Any, method: str, event_dict: EventDict) -> EventDict:
    """
    Filter out “internal” attrs (beginning with ``_``) for console logging.
    """

    to_del = [k for k in event_dict.keys() if k.startswith("_")]
    for k in to_del:
        del event_dict[k]

    return event_dict


def format_timestamp(logger: Any, method: str, event_dict: EventDict) -> EventDict:
    """
    Reformat UNIX timestamps.
    """

    if "timestamp" in event_dict:
        stamp = datetime.fromtimestamp(event_dict["timestamp"])
        event_dict = dict(event_dict)
        event_dict["timestamp"] = stamp.isoformat(timespec="seconds")
        return event_dict
    else:
        return event_dict


def add_process_info(logger: Any, method: str, event_dict: EventDict) -> EventDict:
    """
    Add process info if it does not exist.
    """

    proc = mp.current_process()
    if "pid" not in event_dict:
        event_dict["pid"] = proc.pid
    if "pname" not in event_dict:
        event_dict["pname"] = proc.name

    return event_dict


def log_warning(message, category, filename, lineno, file=None, line=None):
    log = structlog.stdlib.get_logger("lenskit")
    log = log.bind(category=category.__name__, file=filename, lineno=lineno)
    message = str(message)
    m = _WARN_CODE_RE.search(message)
    if m:
        message = message[: m.start()]
        code = m.group(1)
        log.warning(message, err_code=code)
    else:
        log.warning(message)
