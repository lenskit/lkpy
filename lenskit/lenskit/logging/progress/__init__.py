from functools import partial
from typing import Any, Callable, Literal, overload

from ._base import Progress

__all__ = ["Progress", "set_progress_impl"]

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

        case "none" | None:
            _backend = Progress
        case _:
            raise ValueError(f"unknown progress backend {name}")
