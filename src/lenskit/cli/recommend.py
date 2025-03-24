# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import pickle
import sys
from pathlib import Path

import click

import lenskit.operations as ops
from lenskit.data import Dataset
from lenskit.logging import get_logger
from lenskit.random import random_generator

_log = get_logger(__name__)


@click.command("recommend")
@click.option(
    "-o",
    "--output",
    "out_file",
    metavar="FILE",
    help="Output file for recommendations.",
)
@click.option("-n", "--list-length", type=int, help="Recommendation list length.")
@click.option("-d", "--dataset", metavar="DATA", type=Path, help="Use dataset DATA.")
@click.option("-u", "--users-file", type=Path, metavar="FILE", help="Load list of users from FILE.")
@click.option("--random-users", type=int, metavar="N", help="Recommend for N random users.")
@click.argument("PIPE_FILE", type=Path)
@click.argument("USERS", nargs=-1)
def recommend(
    out_file: Path,
    users_file: Path | None,
    random_users: int | None,
    list_length: int | None,
    dataset: Path | None,
    pipe_file: Path,
    users: list,
):
    """
    Generate recommendations from a serialized recommendation pipeline.
    """
    _log.warning("the recommend CLI is experimental and may change without notice")

    _log.info("loading pipeline", file=str(pipe_file))
    with open(pipe_file, "rb") as pf:
        pipe = pickle.load(pf)
    log = _log.bind(name=pipe.name)

    if dataset is not None:
        data = Dataset.load(dataset)
        log = log.bind(data=data.name)
    else:
        data = None

    if random_users is not None:
        if data is None:
            log.error("dataset required for random users")
            sys.exit(5)
        rng = random_generator()
        log.info("selecting random users", count=random_users)
        users = rng.choice(data.users.ids(), random_users)  # type: ignore

    for user in users:
        ulog = log.bind(user=user)
        ulog.debug("generating single-user recommendations")
        recs = ops.recommend(pipe, user, list_length)
        ulog.info("recommended for user", length=len(recs))

        titles = None
        if data is not None:
            items = data.entities("item")
            if "title" in items.attributes:
                titles = items.select(ids=recs.ids()).attribute("title").pandas()

        for item in recs.ids():
            if titles is not None:
                print("item {}: {}".format(item, titles.loc[item]))
            else:
                print("item {}".format(item))
