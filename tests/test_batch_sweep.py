import pathlib

import pandas as pd
import numpy as np

from lk_test_utils import ml_pandas, norm_path

from lenskit import batch, sharing, crossfold as xf
from lenskit.algorithms.basic import Bias, Popular

from pytest import mark

share_impls = [None] + sharing.share_impls


@mark.parametrize('share', share_impls)
def test_sweep_bias(tmp_path, share):
    tmp_path = norm_path(tmp_path)
    work = pathlib.Path(tmp_path)
    ctx = share() if share else None
    sweep = batch.MultiEval(tmp_path, share_context=ctx, nprocs=2)

    ratings = ml_pandas.renamed.ratings
    folds = xf.partition_users(ratings, 5, xf.SampleN(5))
    sweep.add_datasets(folds, DataSet='ml-small')
    sweep.add_algorithms([Bias(damping=0), Bias(damping=5), Bias(damping=10)],
                         attrs=['damping'])
    sweep.add_algorithms(Popular())

    try:
        sweep.run()
    finally:
        if (work / 'runs.csv').exists():
            runs = pd.read_csv(work / 'runs.csv')
            print(runs)

    assert (work / 'runs.csv').exists()
    assert (work / 'runs.parquet').exists()
    assert (work / 'predictions.parquet').exists()
    assert (work / 'recommendations.parquet').exists()

    runs = pd.read_parquet(work / 'runs.parquet')
    # 4 algorithms by 5 partitions
    assert len(runs) == 20
    assert all(np.sort(runs.AlgoClass.unique()) == ['Bias', 'Popular'])
    bias_runs = runs[runs.AlgoClass == 'Bias']
    assert all(bias_runs.damping.notna())
    pop_runs = runs[runs.AlgoClass == 'Popular']
    assert all(pop_runs.damping.isna())

    preds = pd.read_parquet(work / 'predictions.parquet')
    assert all(preds.RunId.isin(bias_runs.RunId))

    recs = pd.read_parquet(work / 'recommendations.parquet')
    assert all(recs.RunId.isin(runs.RunId))
