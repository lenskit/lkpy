from __future__ import annotations

from typing import Any


class Progress:
    """
    Base class for progress reporting.  The default implementations do nothing.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    def update(self, advance: int = 0, **kwargs: float | int | str):
        """
        Update the progress bar.
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

    def __exit__(self, *args):
        self.finish()
