# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from typing import Any, Callable

from typing_extensions import override

from .dataset import Dataset
from .items import ItemList
from .vocab import Vocabulary


class LazyDataset(Dataset):
    """
    A data set with an underlying load function, that doesn't call the function
    until data is actually needed.

    Args:
        loader:
            The function that will load the dataset when needed.
    """

    _delegate: Dataset | None = None
    _loader: Callable[[], Dataset]

    def __init__(self, loader: Callable[[], Dataset]):
        """
        Construct a lazy dataset.
        """
        self._loader = loader

    def delegate(self) -> Dataset:
        """
        Get the delegate data set, loading it if necessary.
        """
        if self._delegate is None:
            self._delegate = self._loader()
        return self._delegate

    @property
    @override
    def items(self) -> Vocabulary:
        return self.delegate().items

    @property
    @override
    def users(self) -> Vocabulary:
        return self.delegate().users

    @override
    def count(self, what: str) -> int:
        return self.delegate().count(what)

    @override
    def interaction_matrix(self, *args, **kwargs) -> Any:
        return self.delegate().interaction_matrix(*args, **kwargs)

    @override
    def interaction_log(self, *args, **kwargs) -> Any:
        return self.delegate().interaction_log(*args, **kwargs)

    @override
    def user_row(self, *args, **kwargs) -> ItemList | None:
        return self.delegate().user_row(*args, **kwargs)
