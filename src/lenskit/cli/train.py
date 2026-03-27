# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import pickle
from pathlib import Path

import click
import torch
from xopen import xopen

from lenskit.data import Dataset
from lenskit.logging import get_logger
from lenskit.pipeline import Component, Pipeline, topn_pipeline
from lenskit.pipeline._types import import_path_string
from lenskit.training import TrainingOptions

_log = get_logger(__name__)


@click.command("train")
@click.option("-C", "--scorer-class", metavar="CLS", help="Train a model with scorer CLS.")
@click.option(
    "-c", "--config", type=Path, metavar="FILE", help="Load pipeline configuration from FILE."
)
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
@click.option("--profile-torch", is_flag=True, help="Profile PyTorch training")
@click.argument("dataset", metavar="DATA", type=Path)
def train(
    scorer_class: str | None,
    config: Path | None,
    out_file: Path,
    name: str | None,
    list_length: int | None,
    profile_torch: bool,
    rating_predictor: bool,
    dataset: Path,
):
    """
    Train a recommendation pipeline and serialize it to disk.
    """
    _log.warning("the training CLI is experimental and may change without notice")

    if config is not None:
        if scorer_class is not None:
            _log.error("cannot specify both scorer class and configuration file")
            raise SystemExit(3)

        pipe = Pipeline.load_config(config)

    elif scorer_class is not None:
        scorer = import_path_string(scorer_class)
        if name is None:
            name = scorer.__name__
        assert issubclass(scorer, Component)
        pipe = topn_pipeline(scorer, predicts_ratings=rating_predictor, n=list_length, name=name)

    else:
        _log.error("no scorer specified")
        raise SystemExit(5)

    data = Dataset.load(dataset)
    log = _log.bind(data=data.name, name=name)

    if profile_torch:  # pragma: nocover
        log.info("setting up Torch profiler")

        try:
            with torch.profiler.profile(with_stack=True) as prof:
                log.info("training pipeline")
                pipe.train(data, TrainingOptions(torch_profiler=prof))
        finally:
            log.info("collecting profile data")
            prof_data = prof.key_averages()
            if torch.cuda.is_available():
                tbl = prof_data.table("self_cuda_time_total")
            else:
                tbl = prof_data.table("self_cpu_time_total")
            print(tbl)
    else:
        log.info("training pipeline")
        pipe.train(data)

    _log.info("saving trained model", file=out_file)
    with xopen(out_file, "wb", threads=0) as pf:
        pickle.dump(pipe, pf)
