from os import fspath
import pathlib
import json
import pickle
import gzip

import pandas as pd
import numpy as np

from lenskit.util.test import ml_test
from lenskit import batch, crossfold as xf
from lenskit.algorithms import Predictor
from lenskit.algorithms.basic import Bias, Popular, TopN

from pytest import mark


@mark.slow
@mark.parametrize('ncpus', [None, 2])
def test_sweep_bias(tmp_path, ncpus):
    work = pathlib.Path(tmp_path)
    sweep = batch.MultiEval(tmp_path, eval_n_jobs=ncpus)

    ratings = ml_test.ratings
    folds = xf.partition_users(ratings, 5, xf.SampleN(5))
    sweep.add_datasets(folds, DataSet='ml-small')
    sweep.add_algorithms(Popular())
    sweep.add_algorithms([Bias(damping=0), Bias(damping=5), Bias(damping=10)],
                         attrs=['damping'])

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
    assert recs['score'].dtype == np.float64


@mark.slow
def test_sweep_norecs(tmp_path):
    work = pathlib.Path(tmp_path)
    sweep = batch.MultiEval(tmp_path, recommend=None)

    ratings = ml_test.ratings
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
    assert not (work / 'recommendations.parquet').exists()

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


@mark.slow
def test_sweep_nopreds(tmp_path):
    work = pathlib.Path(tmp_path)
    sweep = batch.MultiEval(tmp_path, eval_n_jobs=2)

    ratings = ml_test.ratings
    folds = [(train, test.drop(columns=['rating']))
             for (train, test) in xf.partition_users(ratings, 5, xf.SampleN(5))]
    sweep.add_datasets(folds, DataSet='ml-small')
    sweep.add_algorithms(Popular())
    sweep.add_algorithms([Bias(damping=0), Bias(damping=5), Bias(damping=10)],
                         attrs=['damping'])

    try:
        sweep.run()
    finally:
        if (work / 'runs.csv').exists():
            runs = pd.read_csv(work / 'runs.csv')
            print(runs)

    assert (work / 'runs.csv').exists()
    assert (work / 'runs.parquet').exists()
    assert not (work / 'predictions.parquet').exists()
    assert (work / 'recommendations.parquet').exists()

    runs = pd.read_parquet(work / 'runs.parquet')
    # 4 algorithms by 5 partitions
    assert len(runs) == 20
    assert all(np.sort(runs.AlgoClass.unique()) == ['Bias', 'Popular'])
    bias_runs = runs[runs.AlgoClass == 'Bias']
    assert all(bias_runs.damping.notna())
    pop_runs = runs[runs.AlgoClass == 'Popular']
    assert all(pop_runs.damping.isna())

    recs = pd.read_parquet(work / 'recommendations.parquet')
    assert all(recs.RunId.isin(runs.RunId))
    assert recs['score'].dtype == np.float64


@mark.slow
def test_sweep_allrecs(tmp_path):
    work = pathlib.Path(tmp_path)
    sweep = batch.MultiEval(tmp_path, recommend=True)

    ratings = ml_test.ratings
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


@mark.slow
def test_sweep_filenames(tmp_path):
    work = pathlib.Path(tmp_path)
    sweep = batch.MultiEval(tmp_path)

    ratings = ml_test.ratings
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

    def progress(iter, total=None):
        assert total == len(folds) * 4
        return iter

    try:
        sweep.run(progress=progress)
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


@mark.slow
def test_sweep_persist(tmp_path):
    work = pathlib.Path(tmp_path)
    sweep = batch.MultiEval(tmp_path)

    ratings = ml_test.ratings
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

    assert (work / 'runs.csv').exists()
    assert (work / 'runs.parquet').exists()
    assert (work / 'predictions.parquet').exists()
    assert (work / 'recommendations.parquet').exists()

    runs = pd.read_parquet(work / 'runs.parquet')
    # 4 algorithms by 5 partitions
    assert len(runs) == 20


@mark.slow
def test_sweep_oneshot(tmp_path):
    work = pathlib.Path(tmp_path)
    sweep = batch.MultiEval(tmp_path, combine=False)

    ratings = ml_test.ratings
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


@mark.slow
def test_sweep_save(tmp_path):
    work = pathlib.Path(tmp_path)
    sweep = batch.MultiEval(tmp_path)

    ratings = ml_test.ratings
    sweep.add_datasets(lambda: xf.partition_users(ratings, 5, xf.SampleN(5)), name='ml-small')
    sweep.add_algorithms(Bias(damping=5))

    sweep.persist_data()
    pf = work / 'sweep.dat'
    with pf.open('wb') as f:
        pickle.dump(sweep, f)

    with pf.open('rb') as f:
        sweep = pickle.load(f)

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
    # 1 algorithms by 5 partitions
    assert len(runs) == 5


@mark.slow
def test_sweep_combine(tmp_path):
    work = pathlib.Path(tmp_path)
    sweep = batch.MultiEval(tmp_path, combine=False)

    ratings = ml_test.ratings
    sweep.add_datasets(lambda: xf.partition_users(ratings, 5, xf.SampleN(5)), name='ml-small')

    sweep.add_algorithms([Bias(damping=0), Bias(damping=5)],
                         attrs=['damping'])
    sweep.add_algorithms(Popular())

    sweep.persist_data()

    for i in range(1, 6):
        assert (work / 'ds{}-train.parquet'.format(i)).exists()
        assert (work / 'ds{}-test.parquet'.format(i)).exists()

    for ds, cf, dsa in sweep.datasets:
        assert isinstance(ds, tuple)
        train, test = ds
        assert isinstance(train, pathlib.Path)
        assert isinstance(test, pathlib.Path)

    assert sweep.run_count() == 5 * 3

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

    for i, (ds, a) in enumerate(sweep._flat_runs()):
        run = i + 1
        assert (work / 'run-{}.json'.format(run)).exists()
        if isinstance(a.algorithm, Predictor):
            assert (work / 'predictions-{}.parquet'.format(run)).exists()
        assert (work / 'recommendations-{}.parquet'.format(run)).exists()

    sweep.collect_results()

    assert (work / 'runs.csv').exists()
    assert (work / 'runs.parquet').exists()
    assert (work / 'predictions.parquet').exists()
    assert (work / 'recommendations.parquet').exists()

    runs = pd.read_parquet(work / 'runs.parquet')
    assert len(runs) == 5 * 3


@mark.slow
@mark.parametrize("format", [True, 'gzip'])
def test_save_models(tmp_path, format):
    work = pathlib.Path(tmp_path)
    sweep = batch.MultiEval(tmp_path, save_models=format)

    sweep.add_algorithms(Bias(5))
    sweep.add_algorithms(Popular())

    ratings = ml_test.ratings
    sweep.add_datasets(lambda: xf.sample_users(ratings, 2, 100, xf.SampleN(5)),
                       name='ml-small')

    sweep.run()

    runs = pd.read_parquet(fspath(tmp_path / 'runs.parquet'))
    runs = runs.set_index('RunId')

    for i in range(4):
        run_id = i + 1
        fn = work / 'model-{}'.format(run_id)
        if format is True:
            fn = fn.with_suffix('.pkl')
            assert fn.exists()
            with fn.open('rb') as f:
                algo = pickle.load(f)

        elif format == 'gzip':
            fn = fn.with_suffix('.pkl.gz')
            assert fn.exists()
            with gzip.open(fspath(fn), 'rb') as f:
                algo = pickle.load(f)
        else:
            assert False

        assert algo is not None
        algo_class = algo.__class__.__name__
        if isinstance(algo, TopN):
            algo_class = algo.predictor.__class__.__name__

        assert algo_class == runs.loc[run_id, 'AlgoClass']
