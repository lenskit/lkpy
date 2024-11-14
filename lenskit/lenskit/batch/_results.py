from __future__ import annotations

from lenskit.data import EntityId


class BatchResults:
    """
    Results from a batch recommendation run.  Results consist of the outputs of
    various pipeline components for each of the test users.  Results may be
    ``None``, if the pipeline produced no output for that user.
    """

    _data: dict[str, dict[EntityId, object]]

    def __init__(self):
        """
        Construct a new set of batch results.
        """
        self._data = {}

    @property
    def outputs(self) -> list[str]:
        """
        Get the list of output names in these results.
        """
        return list(self._data.keys())

    def output(self, name: str) -> dict[EntityId, object]:
        """
        Get the item lists for a particular output component.

        Args:
            name:
                The output name. This may or may not be the same as the
                component name.
        """
        return self._data[name]

    def add_result(self, name: str, user: EntityId, result: object):
        """
        Add a single result for one of the outputs.

        Args:
            name:
                The output name in which to save this result.
            user:
                The user identifier for this result.
            result:
                The result object to save.
        """

        if name not in self._data:
            self._data[name] = {}

        self._data[name][user] = result
