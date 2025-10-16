# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2025 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

import numpy as np
import ray

from lenskit.pipeline import PipelineConfig
from lenskit.splitting import TTSplit

from .spec import TuningSpec


@dataclass
class TuningJobData:
    """
    Data and configuration for a tuning run.
    """

    spec: TuningSpec
    random_seed: np.random.SeedSequence
    data_name: str | None

    factory_ref: ray.ObjectRef | None = None
    data_ref: ray.ObjectRef | None = None

    @property
    def pipeline(self) -> PipelineConfig:
        assert isinstance(self.spec.pipeline, PipelineConfig)
        return self.spec.pipeline

    @property
    def data(self) -> TTSplit:
        assert self.data_ref is not None
        return ray.get(self.data_ref)
