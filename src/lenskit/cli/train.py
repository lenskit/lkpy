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
from lenskit.training import TrainingOptions

from .pipeline import PipelineLoadSpec, wants_pipeline_config

_log = get_logger(__name__)


@click.command("train")
@click.option("--profile-torch", is_flag=True, help="Profile PyTorch training")
@click.argument("dataset", metavar="DATA", type=Path)
@wants_pipeline_config
def train(
    pipe_cfg: PipelineLoadSpec,
    out_file: Path,
    profile_torch: bool,
    dataset: Path,
):
    """
    Train a recommendation pipeline and serialize it to disk.
    """
    _log.warning("the training CLI is experimental and may change without notice")

    pipe = pipe_cfg.load_pipeline()

    data = Dataset.load(dataset)
    log = _log.bind(data=data.name, name=pipe.name)

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
