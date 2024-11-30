from functools import partial
from typing import Any, Callable, Literal, overload

from ._base import Progress

_backend: Callable[..., Progress] = Progress


@overload
def set_progress_impl(name: Literal["tqdm"], impl: Callable[..., Any] | None = None, /): ...
@overload
def set_progress_impl(name: Literal["rich"]): ...
def set_progress_impl(name: str | None, *options: Any):
    global _backend

    match name:
        case "tqdm":
            from ._tqdm import TQDMProgress, tqdm

            impl = tqdm
            if options and options[0]:
                impl = options[0]

            _backend = partial(TQDMProgress, impl)

        case "rich":
            from ._rich import RichProgress

            _backend = RichProgress

        case "none" | None:
            _backend = Progress
        case _:
            raise ValueError(f"unknown progress backend {name}")


def item_progress(label: str, total: int, fields: dict[str, str | None] | None = None) -> Progress:
    """
    Create a progress bar for distinct, counted items.

    Args:
        label:
            The progress bar label.
        total:
            The total number of items.
        fields:
            Additional fields to report with the progress bar (such as a current
            loss).  These are specified as a dictionary mapping field names to
            format strings (the pieces inside ``{...}`` in :meth:`str.format`),
            and the values come from extra kwargs to :meth:`Progress.update`;
            mapping to ``None`` use default ``str`` formatting.
    """
    return _backend(label, total, fields)
