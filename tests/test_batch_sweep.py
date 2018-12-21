import pathlib
import json

import pandas as pd
import numpy as np

from lk_test_utils import ml_pandas, norm_path

from lenskit import batch, crossfold as xf
from lenskit.algorithms.basic import Bias, Popular

from pytest import mark


@mark.parametrize('ncpus', [None, 2])
def test_sweep_bias(tmp_path, ncpus):
    tmp_path = norm_path(tmp_path)
    work = pathlib.Path(tmp_path)
    sweep = batch.MultiEval(tmp_path, nprocs=ncpus)

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


def test_sweep_filenames(tmp_path):
    tmp_path = norm_path(tmp_path)
    work = pathlib.Path(tmp_path)
    sweep = batch.MultiEval(tmp_path)

    ratings = ml_pandas.renamed.ratings
    folds = []
    for part, (train, test) in enumerate(xf.partition_users(ratings, 2, xf.SampleN(5))):
        trfn = work / 'p{}-train.csv'.format(part)
        tefn = work / 'p{}-test.csv'.format(part)
        train.to_csv(trfn)
        test.to_csv(tefn)
        folds.append((trfn, tefn))

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
    # 4 algorithms by 2 partitions
    assert len(runs) == 8


def test_sweep_persist(tmp_path):
    tmp_path = norm_path(tmp_path)
    work = pathlib.Path(tmp_path)
    sweep = batch.MultiEval(tmp_path)

    ratings = ml_pandas.renamed.ratings
    sweep.add_datasets(lambda: xf.partition_users(ratings, 5, xf.SampleN(5)), name='ml-small')
    sweep.persist_data()

    for i in range(1,6):
        assert (work / 'ds{}-train.parquet'.format(i)).exists()
        assert (work / 'ds{}-test.parquet'.format(i)).exists()

    for ds, cf, dsa in sweep.datasets:
        assert isinstance(ds, tuple)
        train, test = ds
        assert isinstance(train, pathlib.Path)
        assert isinstance(test, pathlib.Path)

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


def test_sweep_oneshot(tmp_path):
    tmp_path = norm_path(tmp_path)
    work = pathlib.Path(tmp_path)
    sweep = batch.MultiEval(tmp_path, combine=False)

    ratings = ml_pandas.renamed.ratings
    sweep.add_datasets(lambda: xf.partition_users(ratings, 5, xf.SampleN(5)), name='ml-small')
    sweep.add_algorithms(Bias(damping=5))

    try:
        sweep.run(3)
    finally:
        if (work / 'runs.csv').exists():
            runs = pd.read_csv(work / 'runs.csv')
            print(runs)

    assert not (work / 'runs.csv').exists()
    assert not (work / 'runs.parquet').exists()
    assert not (work / 'predictions.parquet').exists()
    assert not (work / 'recommendations.parquet').exists()

    assert (work / 'run-3.json').exists()
    assert (work / 'predictions-3.parquet').exists()
    assert (work / 'recommendations-3.parquet').exists()

    with (work / 'run-3.json').open() as f:
        run = json.load(f)
    assert run['RunId'] == 3


def test_sweep_combine(tmp_path):
    tmp_path = norm_path(tmp_path)
    work = pathlib.Path(tmp_path)
    sweep = batch.MultiEval(tmp_path, combine=False)

    ratings = ml_pandas.renamed.ratings
    sweep.add_datasets(lambda: xf.partition_users(ratings, 5, xf.SampleN(5)), name='ml-small')
    sweep.persist_data()

    for i in range(1, 6):
        assert (work / 'ds{}-train.parquet'.format(i)).exists()
        assert (work / 'ds{}-test.parquet'.format(i)).exists()

    for ds, cf, dsa in sweep.datasets:
        assert isinstance(ds, tuple)
        train, test = ds
        assert isinstance(train, pathlib.Path)
        assert isinstance(test, pathlib.Path)

    sweep.add_algorithms([Bias(damping=0), Bias(damping=5), Bias(damping=10)],
                         attrs=['damping'])
    sweep.add_algorithms(Popular())

    try:
        sweep.run()
    finally:
        if (work / 'runs.csv').exists():
            runs = pd.read_csv(work / 'runs.csv')
            print(runs)

    assert not (work / 'runs.csv').exists()
    assert not (work / 'runs.parquet').exists()
    assert not (work / 'predictions.parquet').exists()
    assert not (work / 'recommendations.parquet').exists()

    for i in range(1, 6):
        assert (work / 'run-{}.json'.format(i)).exists()
        assert (work / 'predictions-{}.parquet'.format(i)).exists()
        assert (work / 'recommendations-{}.parquet'.format(i)).exists()

    sweep.collect_results()

    assert (work / 'runs.csv').exists()
    assert (work / 'runs.parquet').exists()
    assert (work / 'predictions.parquet').exists()
    assert (work / 'recommendations.parquet').exists()

    runs = pd.read_parquet(work / 'runs.parquet')
    # 4 algorithms by 5 partitions
    assert len(runs) == 20
