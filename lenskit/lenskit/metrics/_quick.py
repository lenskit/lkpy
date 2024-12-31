from __future__ import annotations

import logging

from lenskit.data import Dataset
from lenskit.pipeline import Component, RecPipelineBuilder
from lenskit.types import RNGInput

from .bulk import RunAnalysis, RunAnalysisResult
from .predict import MAE, RMSE
from .ranking import NDCG, RBP, Hit, Recall, RecipRank

_log = logging.getLogger(__name__)


def quick_measure_model(
    model: Component,
    data: Dataset,
    *,
    predicts_ratings: bool = False,
    n_jobs: int | None = 1,
    rng: RNGInput = None,
) -> RunAnalysisResult:
    """
    Do a quick-and-dirty model measurement with a default pipeline setup, split,
    and metrics. This is mostly to make tests easy to write, you usually don't
    want to use it for actual recommender evaluation.

    Stability:
        Caller
    """
    from lenskit.basic import BiasScorer
    from lenskit.batch import BatchPipelineRunner
    from lenskit.splitting import SampleFrac, sample_users

    builder = RecPipelineBuilder()
    builder.scorer(model)
    if predicts_ratings:
        builder.predicts_ratings(fallback=BiasScorer())

    pipe = builder.build()

    n_users = data.user_count
    us_size = n_users // 5
    split = sample_users(data, us_size, SampleFrac(0.2, rng=rng), rng=rng)
    _log.info("measuring %s on %d users", model, us_size)

    pipe.train(split.train)
    runner = BatchPipelineRunner(n_jobs=n_jobs)
    runner.recommend(n=20)
    if predicts_ratings:
        runner.predict()

    outs = runner.run(pipe, split.test)

    rra = RunAnalysis()
    rra.add_metric(RecipRank())
    rra.add_metric(RBP())
    rra.add_metric(NDCG())
    rra.add_metric(Hit())
    rra.add_metric(Recall())

    result = rra.compute(outs.output("recommendations"), split.test)

    if predicts_ratings:
        pra = RunAnalysis()
        pra.add_metric(RMSE())
        pra.add_metric(MAE())
        pr = pra.compute(outs.output("predictions"), split.test)
        result.merge_from(pr)

    return result
