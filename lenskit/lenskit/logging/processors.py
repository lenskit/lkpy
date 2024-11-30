"""
LensKit logging processors.
"""

from datetime import datetime
from typing import Any

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
        return event_dict | {"timestamp": stamp.isoformat(timespec="seconds")}
    else:
        return event_dict
