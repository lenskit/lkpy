# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Hyperparameter search.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import click
import ray.tune.utils.log
from humanize import metric as human_metric
from humanize import precisedelta
from pydantic_core import to_json

from lenskit.logging import get_logger, stdout_console
from lenskit.parallel.ray import init_cluster
from lenskit.tuning import PipelineTuner, TuningSpec

_log = get_logger(__name__)


@click.command("tune")
@click.option(
    "-T", "--training-data", type=Path, required=True, help="path to the training dataset"
)
@click.option(
    "-V",
    "--valid-data",
    "--tuning-data",
    "tuning_data",
    type=Path,
    required=True,
    help="path to the tuning/validation data",
)
@click.option("--random", "method", flag_value="random", help="use random search")
@click.option(
    "--hyperopt", "method", flag_value="hyperopt", help="use HyperOpt search (requires Ray)"
)
@click.option(
    "--optuna",
    "method",
    flag_value="optuna",
    default="optuna",
    help="use Optuna TPE search (default)",
)
@click.option(
    "--ray", "use_ray", is_flag=True, help="use Ray Tune instead of the direct Optuna search"
)
@click.option("-j", "--job-limit", type=int, help="limit for concurrent tuning jobs")
@click.option(
    "--max-points", type=int, metavar="N", help="maximum number of configurations to test"
)
@click.option(
    "-m",
    "--metric",
    type=click.Choice(["RMSE", "RBP", "RecipRank", "NDCG"]),
    default="RBP",
    help="the metric to optimize",
)
@click.option(
    "--save-pipeline", type=Path, metavar="FILE", help="Save best pipeline configuration to FILE"
)
@click.argument("SEARCH_SPEC", type=Path)
@click.argument("OUT", type=Path)
def tune(
    *,
    search_spec: Path,
    out: Path,
    method: Literal["random", "hyperopt", "tpe"] | None,
    use_ray: bool,
    max_points: int | None,
    job_limit: int | None,
    metric: str,
    training_data: Path,
    tuning_data: Path,
    save_pipeline: Path | None,
):
    """
    Tune pipeline hyperparameters with Ray Tune.
    """
    console = stdout_console()
    os.environ["RAY_AIR_NEW_OUTPUT"] = "0"
    ray.tune.utils.log.set_verbosity(0)

    spec = TuningSpec.load(search_spec)
    # override settings from command line
    spec.search.update_max_points(max_points)
    if metric is not None:
        spec.search.metric = metric
    if method is not None:
        spec.search.method = method

    out.mkdir(exist_ok=True, parents=True)

    # set up the tuning controller
    if use_ray:
        from lenskit.tuning import RayPipelineTuner

        try:
            ray.init(address="auto")
            _log.info("connected to existing Ray cluster")
        except ConnectionError:
            # use global parallel setup for tuning
            init_cluster(global_logging=True)

        controller = RayPipelineTuner(spec, out)
    else:
        controller = PipelineTuner(spec, out)

    if job_limit is not None:
        controller.settings.jobs = job_limit
    controller.set_data(training_data, tuning_data)

    with open(out / "config.json", "wt") as jsf:
        print(controller.spec.model_dump_json(indent=2), file=jsf)

    _log.info("starting hyperparameter search")
    results = controller.run()

    best = results.best_result()
    result_file = out / "best-result.json"
    result_json = to_json(best, indent=2)
    result_file.write_bytes(result_json + b"\n")

    best_cfg = results.best_config()
    cfg_file = out / "best-result.json"
    cfg_json = to_json(best_cfg, indent=2)
    cfg_file.write_bytes(cfg_json + b"\n")

    if save_pipeline is not None:
        pipe_json = results.best_pipeline().model_dump_json(indent=2)
        save_pipeline.write_text(pipe_json + "\n")

    _log.info("finished hyperparameter search")
    console.print("[bold yellow]Hyperparameter search completed![/bold yellow]")
    console.print("Best {} is [bold red]{:.3f}[/bold red]".format(metric, best[metric]))
    assert results.task.duration is not None
    line = "[bold magenta]{}[/bold magenta] trials took [bold cyan]{}[/bold cyan]".format(
        results.num_trials(),
        precisedelta(results.task.duration),  # type: ignore
    )
    if results.task.system_power:
        line += " and consumed [bold green]{}[/bold green]".format(
            human_metric(results.task.system_power / 3600, unit="Wh")
        )
    console.print(line)
    console.print("Trial result:", best)
