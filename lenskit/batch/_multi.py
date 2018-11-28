import logging
import pathlib
import collections
import warnings

import pandas as pd

from ..algorithms import Predictor
from .. import topn, util
from ._recommend import recommend
from ._predict import predict

try:
    import fastparquet
except ImportError:
    fastparquet = None

_logger = logging.getLogger(__name__)

_AlgoRec = collections.namedtuple('_AlgoRec', [
    'algorithm',
    'parallel',
    'attributes'
])
_DSRec = collections.namedtuple('_DSRec', [
    'dataset',
    'candidates',
    'attributes'
])


class MultiEval:
    """
    A runner for carrying out multiple evaluations, such as parameter sweeps.

    Args:
        path(str or :py:class:`pathlib.Path`):
            the working directory for this evaluation.
            It will be created if it does not exist.
        predict(bool):
            whether to generate rating predictions.
        recommend(int):
            the number of recommendations to generate per user (None to disable top-N).
        candidates(function):
            the default candidate set generator for recommendations.  It should take the
            training data and return a candidate generator, itself a function mapping user
            IDs to candidate sets.
    """

    def __init__(self, path, predict=True,
                 recommend=100, candidates=topn.UnratedCandidates, nprocs=None):
        self.workdir = pathlib.Path(path)
        self.predict = predict
        self.recommend = recommend
        self.candidate_generator = candidates
        self.algorithms = []
        self.datasets = []
        self.nprocs = nprocs

    @property
    def run_csv(self):
        return self.workdir / 'runs.csv'

    @property
    def run_file(self):
        return self.workdir / 'runs.parquet'

    @property
    def preds_file(self):
        return self.workdir / 'predictions.parquet'

    @property
    def recs_file(self):
        return self.workdir / 'recommendations.parquet'

    def add_algorithms(self, algos, parallel=False, attrs=[], **kwargs):
        """
        Add one or more algorithms to the run.

        Args:
            algos(algorithm or list): the algorithm(s) to add.
            parallel(bool):
                if ``True``, allow this algorithm to be trained in parallel with others.
            attrs(list of str):
                a list of attributes to extract from the algorithm objects and include in
                the run descriptions.
            kwargs:
                additional attributes to include in the run descriptions.
        """

        if not isinstance(algos, collections.Iterable):
            algos = [algos]

        for algo in algos:
            aa = {'AlgoClass': algo.__class__.__name__, 'AlgoStr': str(algo)}
            aa.update(kwargs)
            for an in attrs:
                aa[an] = getattr(algo, an, None)

            self.algorithms.append(_AlgoRec(algo, parallel, aa))

    def add_datasets(self, data, name=None, candidates=None, **kwargs):
        """
        Add one or more datasets to the run.

        Args:
            data:
                The input data set(s) to run. Can be one of the following:

                * A tuple of (train, test) data.
                * An iterable of (train, test) pairs, in which case the iterable
                  is not consumed until it is needed.
                * A function yielding either of the above, to defer data load
                  until it is needed.

                Data can be either data frames or paths; paths are loaded after
                detection using :py:fun:`util.read_df_detect`.

            kwargs:
                additional attributes pertaining to these data sets.
        """

        attrs = {}
        if name is not None:
            attrs['DataSet'] = name
        attrs.update(kwargs)

        self.datasets.append(_DSRec(data, candidates, attrs))

    def persist_data(self):
        """
        Persist the data for an experiment, replacing in-memory data sets with file names.
        Once this has been called, the sweep can be pickled.
        """
        self.workdir.mkdir(parents=True, exist_ok=True)
        ds2 = []
        for i, (ds, cand_f, ds_attrs) in enumerate(self._flat_datasets()):
            train, test = ds
            if isinstance(train, pd.DataFrame):
                fn = self.workdir / 'ds{}-train.parquet'.format(i+1)
                _logger.info('serializing to %s', fn)
                train.to_parquet(fn)
                train = fn
            if isinstance(test, pd.DataFrame):
                fn = self.workdir / 'ds{}-test.parquet'.format(i+1)
                _logger.info('serializing to %s', fn)
                test.to_parquet(fn)
                test = fn
            ds2.append(((train, test), cand_f, ds_attrs))
        self.datasets = ds2

    def _normalize_ds_entry(self, entry):
        # normalize data set to be an iterable of tuples
        ds, cand_f, attrs = entry
        if callable(ds):
            ds = ds()
        if isinstance(ds, tuple):
            yield _DSRec(ds, cand_f, attrs)
        else:
            yield from (_DSRec(dse, cand_f, dict(Partition=part+1, **attrs))
                        for (part, dse) in enumerate(ds))

    def _flat_datasets(self):
        for entry in self.datasets:
            yield from self._normalize_ds_entry(entry)

    def _read_data(self, df):
        if isinstance(df, str) or isinstance(df, pathlib.Path):
            _logger.info('reading from %s', df)
            return util.read_df_detect(df)
        else:
            return df

    def _flat_runs(self):
        for dse in self._flat_datasets():
            for arec in self.algorithms:
                yield (dse, arec)

    def run(self):
        """
        Run the evaluation.
        """

        self.workdir.mkdir(parents=True, exist_ok=True)

        run_id = 0
        run_data = []
        train_load = util.LastMemo(self._read_data)
        test_load = util.LastMemo(self._read_data)

        for i, (dsrec, arec) in enumerate(self._flat_runs()):
            run_id = i + 1

            ds, cand_f, ds_attrs = dsrec
            if cand_f is None:
                cand_f = self.candidate_generator
            train, test = ds
            train = train_load(train)
            test = test_load(train)

            ds_name = ds_attrs.get('DataSet', None)
            ds_part = ds_attrs.get('Partition', None)
            cand = cand_f(train)

            _logger.info('starting run %d: %s on %s:%d', run_id, arec.algorithm,
                         ds_name, ds_part)
            run = self._run_algo(run_id, arec, (train, test, ds_attrs, cand))
            _logger.info('finished run %d: %s on %s:%d', run_id, arec.algorithm,
                         ds_name, ds_part)
            run_data.append(run)
            run_df = pd.DataFrame(run_data)
            # overwrite files to show progress
            run_df.to_csv(self.run_csv)
            run_df.to_parquet(self.run_file, compression=None)

    def _run_algo(self, run_id, arec, data):
        train, test, dsp_attrs, cand = data

        run = {'RunId': run_id}
        run.update(dsp_attrs)
        run.update(arec.attributes)

        model, train_time = self._train_algo(arec.algorithm, train)
        run['TrainTime'] = train_time

        preds, pred_time = self._predict(run_id, arec.algorithm, model, test)
        run['PredTime'] = pred_time
        self._write_results(self.preds_file, preds, append=run_id > 1)

        recs, rec_time = self._recommend(run_id, arec.algorithm, model, test, cand)
        run['RecTime'] = rec_time
        self._write_results(self.recs_file, recs, append=run_id > 1)

        return run

    def _train_algo(self, algo, train):
        watch = util.Stopwatch()
        _logger.info('training algorithm %s on %d ratings', algo, len(train))
        model = algo.train(train)
        watch.stop()
        _logger.info('trained algorithm %s in %s', algo, watch)
        return model, watch.elapsed()

    def _predict(self, rid, algo, model, test):
        if not self.predict:
            return None, None
        if not isinstance(algo, Predictor):
            return None, None

        watch = util.Stopwatch()
        _logger.info('generating %d predictions for %s', len(test), algo)
        preds = predict(algo, model, test, nprocs=self.nprocs)
        watch.stop()
        _logger.info('generated predictions in %s', watch)
        preds['RunId'] = rid
        preds = preds[['RunId', 'user', 'item', 'rating', 'prediction']]
        return preds, watch.elapsed()

    def _recommend(self, rid, algo, model, test, candidates):
        if self.recommend is None:
            return None, None

        watch = util.Stopwatch()
        users = test.user.unique()
        _logger.info('generating recommendations for %d users for %s', len(users), algo)
        recs = recommend(algo, model, users, self.recommend, candidates, test,
                         nprocs=self.nprocs)
        watch.stop()
        _logger.info('generated recommendations in %s', watch)
        recs['RunId'] = rid
        return recs, watch.elapsed()

    def _write_results(self, file, df, append=True):
        if df is None:
            return

        if fastparquet is not None:
            fastparquet.write(str(file), df, append=append, compression='snappy')
        elif append and file.exists():
            warnings.warn('fastparquet not available, appending is slow')
            odf = pd.read_parquet(str(file))
            pd.concat([odf, df]).to_parquet(str(file))
        else:
            df.to_parquet(str(file))
