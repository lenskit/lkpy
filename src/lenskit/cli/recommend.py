# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import pickle
import sys
from pathlib import Path

import click
from xopen import xopen

import lenskit.operations as ops
from lenskit import batch
from lenskit.data import Dataset, ItemList, ListILC, UserIDKey
from lenskit.logging import Stopwatch, get_logger, item_progress
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
@click.option("--print/--no-print", "print_recs", default=True, help="Print recommendations.")
@click.option("-n", "--list-length", type=int, help="Recommendation list length.")
@click.option("--batch/--no-batch", "use_batch", default=False, help="Use batch.recommend.")
@click.option("-j", "--process-count", type=int, help="Use specified number of worker processes.")
@click.option("--ray", "use_ray", is_flag=True, help="Ue Ray for parallelism.")
@click.option("-d", "--dataset", metavar="DATA", type=Path, help="Use dataset DATA.")
@click.option("-u", "--users-file", type=Path, metavar="FILE", help="Load list of users from FILE.")
@click.option("--random-users", type=int, metavar="N", help="Recommend for N random users.")
@click.argument("PIPE_FILE", type=Path)
@click.argument("USERS", nargs=-1)
def recommend(
    out_file: Path,
    users_file: Path | None,
    print_recs: bool,
    use_batch: bool,
    process_count: int | None,
    use_ray: bool,
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
    with xopen(pipe_file, "rb", threads=0) as pf:
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

    if use_batch or use_ray or process_count is not None:
        if use_ray:
            n_jobs = "ray"
        else:
            n_jobs = process_count

        all_recs = batch.recommend(pipe, users, list_length, n_jobs=n_jobs)
    else:
        timer = Stopwatch(start=False)
        all_recs = None if out_file is None else ListILC(UserIDKey)
        with item_progress("user recommendations", len(users)) as pb:
            for user in users:
                ulog = log.bind(user=user)
                ulog.debug("generating single-user recommendations")
                with timer.measure(accumulate=True):
                    recs = ops.recommend(pipe, user, list_length)
                ulog.info(
                    "recommended for user",
                    length=len(recs),
                    time="{:.1f}ms".format(timer.elapsed(accumulated=False) * 1000),
                )

                if all_recs is not None:
                    all_recs.add(recs, user_id=user)

                if print_recs:
                    print_recommendation_list(recs, data)

                pb.update()

        log.info(
            "finished recommending for %d users in %s (%.1fms/u)",
            len(users),
            timer,
            timer.elapsed() * 1000 / len(users),
        )

    if out_file is not None:
        assert all_recs is not None
        log.info("saving recommendations to %s", str(out_file), count=len(all_recs))
        all_recs.save_parquet(out_file)


def print_recommendation_list(recs: ItemList, data: Dataset | None):
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
