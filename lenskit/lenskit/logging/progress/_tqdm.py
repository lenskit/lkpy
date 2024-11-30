from __future__ import annotations

from typing import Callable

from tqdm import tqdm

from ._base import Progress

__all__ = ["TQDMProgress", "tqdm"]


class TQDMProgress(Progress):
    """
    TQDM-based progress reporting. Useful for notebooks.
    """

    tqdm: tqdm
    _field_format: str | None = None

    def __init__(
        self,
        cls: Callable[..., tqdm],
        label: str | None,
        total: int | None,
        fields: dict[str, str | None],
    ):
        self.tqdm = tqdm(desc=label, total=total, leave=False)
        if fields:
            self._field_format = ", ".join(
                [f"{name}: {fs or None}" for (name, fs) in fields.items()]
            )

    def update(self, advance: int = 0, **kwargs: float | int | str):
        """
        Update the progress bar.
        """
        self.tqdm.update(advance)
        if self._field_format:
            self.tqdm.set_postfix_str(self._field_format.format(kwargs))

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
