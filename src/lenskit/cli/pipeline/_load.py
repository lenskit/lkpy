# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import click

from lenskit.logging import get_logger
from lenskit.pipeline import Component, Pipeline, topn_pipeline
from lenskit.pipeline._types import import_path_string

_log = get_logger(__name__)


@dataclass
class PipelineLoadSpec:
    scorer_class: str | None = None
    config: Path | None = None
    rating_predictor: bool = False
    list_length: int | None = None
    name: str | None = None

    def load_pipeline(self):
        if self.config is not None:
            if self.scorer_class is not None:
                _log.error("cannot specify both scorer class and configuration file")
                raise SystemExit(3)

            _log.info("loading pipeline from %s", self.config)
            return Pipeline.load_config(self.config)

        elif self.scorer_class is not None:
            _log.info("creating pipeline for class %s", self.scorer_class)
            scorer = import_path_string(self.scorer_class)
            assert issubclass(scorer, Component)
            name = self.name or scorer.__name__
            return topn_pipeline(
                scorer, predicts_ratings=self.rating_predictor, n=self.list_length, name=name
            )

        else:
            _log.error("no scorer specified")
            raise SystemExit(5)


def wants_pipeline_config(func):
    @wraps(func)
    @click.option(
        "-C", "--scorer-class", metavar="CLS", help="Create a default pipeline with scorer CLS"
    )
    @click.option(
        "-c", "--config", type=Path, metavar="FILE", help="Load pipeline configuration from FILE."
    )
    @click.option(
        "--rating-predictor",
        is_flag=True,
        help="Include rating prediction in the pipeline capabilities.",
    )
    @click.option("--pipeline-name", type=str, help="Name of pipeline (when constructing).")
    @click.option("-n", "--list-length", type=int, help="Default list length for pipeline ranker.")
    def wrap_for_config(
        scorer_class, config, rating_predictor, list_length, pipeline_name, **kwargs
    ):
        cfg = PipelineLoadSpec(
            scorer_class=scorer_class,
            config=config,
            rating_predictor=rating_predictor,
            list_length=list_length,
            name=pipeline_name,
        )
        func(cfg, **kwargs)

    return wrap_for_config
