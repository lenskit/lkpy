# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
LensKit logging processors and converters.
"""

import multiprocessing as mp
from datetime import datetime
from typing import Any

import structlog
from structlog.typing import EventDict


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
    log.warning(str(message))
