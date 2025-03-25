# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2025 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import pickle
from pathlib import Path

import click

from lenskit.data import Dataset
from lenskit.logging import get_logger
from lenskit.pipeline import Component, topn_pipeline
from lenskit.pipeline.types import parse_type_string

_log = get_logger(__name__)


@click.command("train")
@click.option("-C", "--scorer-class", metavar="CLS", help="Train a model with scorer CLS.")
@click.option(
    "-o",
    "--output",
    "out_file",
    metavar="FILE",
    default="model.pkl",
    help="Output file for trained model.",
)
@click.option("--name", help="Name of the recommendation pipeline.")
@click.option(
    "--rating-predictor",
    is_flag=True,
    help="Include rating prediction in the pipeline capabilities.",
)
@click.option("-n", "--list-length", type=int, help="Default list length for pipeline ranker.")
@click.argument("dataset", metavar="DATA", type=Path)
def train(
    scorer_class: str | None,
    out_file: Path,
    name: str | None,
    list_length: int | None,
    rating_predictor: bool,
    dataset: Path,
):
    """
    Train a recommendation pipeline and serialize it to disk.
    """
    _log.warning("the training CLI is experimental and may change without notice")

    if scorer_class is not None:
        scorer = parse_type_string(scorer_class)
        if name is None:
            name = scorer.__name__
        assert issubclass(scorer, Component)
        pipe = topn_pipeline(scorer, predicts_ratings=rating_predictor, n=list_length, name=name)
    else:
        _log.error("no scorer specified")
        raise SystemExit(5)

    data = Dataset.load(dataset)
    log = _log.bind(data=data.name, name=name)

    log.info("training model")
    pipe.train(data)

    _log.info("saving trained model", file=out_file)
    with open(out_file, "wb") as pf:
        pickle.dump(pipe, pf)
