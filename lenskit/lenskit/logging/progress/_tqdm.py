from __future__ import annotations

from typing import Callable

from tqdm import tqdm

from ._base import Progress

__all__ = ["TQDMProgress", "tqdm"]


class TQDMProgress(Progress):
    """
    TQDM-based progress reporting. Useful for notebooks.
    """

    def __init__(
        self,
        cls: Callable[..., tqdm],
        label: str | None,
        total: int | None,
        fields: dict[str, str | None],
    ):
        self.tqdm = tqdm(desc=label, total=total, leave=False)

    def update(self, advance: int = 0, **kwargs: float | int | str):
        """
        Update the progress bar.
        """
        self.tqdm.update(advance)

    def finish(self):
        """
        Finish and clean up this progress bar.  If the progresss bar is used as
        a context manager, this is automatically called on context exit.
        """
        self.tqdm.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.finish()
