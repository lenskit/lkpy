"""
Top-level package for the LensKit recommender systems toolkit.
"""

from . import (
    als,
    basic,
    batch,
    config,
    data,
    flexmf,
    funksvd,
    hpf,
    knn,
    logging,
    metrics,
    operations,
    pipeline,
    schemas,
    sklearn,
    splitting,
    stats,
    torch,
    training,
)
from .config import configure, lenskit_config
from .data import Dataset, DatasetBuilder
from .operations import predict, recommend, score
from .pipeline import Component, Pipeline, RecPipelineBuilder, topn_pipeline
from .splitting import TTSplit

__all__ = [
    "Component",
    "Dataset",
    "DatasetBuilder",
    "Pipeline",
    "RecPipelineBuilder",
    "TTSplit",
    "als",
    "basic",
    "batch",
    "config",
    "configure",
    "data",
    "flexmf",
    "funksvd",
    "hpf",
    "knn",
    "lenskit_config",
    "logging",
    "metrics",
    "operations",
    "pipeline",
    "predict",
    "recommend",
    "schemas",
    "score",
    "sklearn",
    "splitting",
    "stats",
    "topn_pipeline",
    "torch",
    "training",
]

__version__: str
