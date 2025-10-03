# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import datetime as dt
import re
from pathlib import Path

import click

from lenskit.data.dataset import Dataset
from lenskit.logging import get_logger
from lenskit.splitting import sample_records, split_global_time, split_temporal_fraction

_log = get_logger(__name__)


@click.command("split")
@click.option("--date", type=str, metavar="DATE", help="split at DATE")
@click.option(
    "--fraction", type=float, metavar="FRAC", help="use FRAC fraction interactions for testing"
)
@click.option("--random", is_flag=True, help="split randomly instead of temporally")
@click.option(
    "--min-train-interactions",
    type=int,
    metavar="N",
    help="drop test users without at least N training interactions",
)
@click.option("-d", "--out-dir", type=Path, help="directory to store split data")
@click.argument("data", required=True, type=Path)
def split(
    data: Path,
    out_dir: Path | None,
    date: str | None,
    fraction: float | None,
    random: bool,
    min_train_interactions: int | None,
):
    """
    Prepare a train-test split of a dataset.

    Note: this command does not expose all of LensKit's data-splitting
    capabilities. Use the `lenskit.splitting` API for more fine-grained control.
    """

    ds = Dataset.load(data)
    _log.info("splitting dataset %s", ds.name)

    if date:
        if re.match(r"^\d+(\.\d+)?$", date):
            split_time = dt.datetime.fromtimestamp(float(date))
        elif re.match(r"^\d+-\d+-\d+$", date):
            split_time = dt.datetime.fromisoformat(date)
        elif re.match(r"^\d+-\d+-\d+[T ]\d+:\d+:\d+(\.\d+)?", date):
            split_time = dt.datetime.fromisoformat(date)
        else:
            _log.error("invalid date %s", date)
            raise click.UsageError("invalid date")

        split = split_global_time(ds, split_time, filter_test_users=min_train_interactions)

    elif fraction:
        if random:
            n = int(ds.interaction_count * fraction)
            split = sample_records(ds, n)
        else:
            split = split_temporal_fraction(ds, fraction, filter_test_users=min_train_interactions)

    else:
        _log.error("no split method specified")
        raise click.UsageError("must specify a split")

    if out_dir is None:
        train_path = data.with_name(data.name + ".train")
        test_path = data.with_name(data.name + ".test.parquet")
    else:
        train_path = out_dir / "train"
        test_path = out_dir / "test.parquet"

    split.train.save(train_path)
    split.test.save_parquet(test_path)
