import logging
import pathlib
import collections
import json

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
        combine(bool):
            whether to combine output; if ``False``, output will be left in separate files, if
            ``True``, it will be in a single set of files (runs, recommendations, and preditions).
    """

    def __init__(self, path, predict=True,
                 recommend=100, candidates=topn.UnratedCandidates,
                 nprocs=None, combine=True):
        self.workdir = pathlib.Path(path)
        self.predict = predict
        self.recommend = recommend
        self.candidate_generator = candidates
        self.algorithms = []
        self.datasets = []
        self.nprocs = nprocs
        self.combine_output = combine
        self._is_flat = True

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

        if not isinstance(data, tuple):
            self._is_flat = False

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
        self._is_flat = True

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

    def run_count(self):
        "Get the number of runs in this evaluation."
        if self._is_flat:
            nds = len(self.datasets)
        else:
            _logger.warning('attempting to count runs in a non-flattened evaluation')
            nds = len(list(self._flat_datasets()))
        return nds * len(self.algorithms)

    def run(self, runs=None):
        """
        Run the evaluation.

        Args:
            runs(int or set-like):
                If provided, a specific set of runs to run.  Useful for splitting
                an experiment into individual runs.  This is a set of 1-based run
                IDs, not 0-based indexes.
        """

        if runs is not None and self.combine_output:
            raise ValueError('Cannot select runs with combined output')

        if runs is not None and not isinstance(runs, collections.Iterable):
            runs = [runs]

        self.workdir.mkdir(parents=True, exist_ok=True)

        run_id = 0
        run_data = []
        train_load = util.LastMemo(self._read_data)
        test_load = util.LastMemo(self._read_data)

        for i, (dsrec, arec) in enumerate(self._flat_runs()):
            run_id = i + 1
            if runs is not None and run_id not in runs:
                _logger.info('skipping deselected run %d', run_id)
                continue

            ds, cand_f, ds_attrs = dsrec
            if cand_f is None:
                cand_f = self.candidate_generator
            train, test = ds
            train = train_load(train)
            test = test_load(test)

            ds_name = ds_attrs.get('DataSet', None)
            ds_part = ds_attrs.get('Partition', None)
            cand = cand_f(train)

            _logger.info('starting run %d: %s on %s:%s', run_id, arec.algorithm,
                         ds_name, ds_part)
            run = self._run_algo(run_id, arec, (train, test, ds_attrs, cand))
            _logger.info('finished run %d: %s on %s:%s', run_id, arec.algorithm,
                         ds_name, ds_part)
            run_data.append(run)
            self._write_run(run, run_data)

    def _run_algo(self, run_id, arec, data):
        train, test, dsp_attrs, cand = data

        run = {'RunId': run_id}
        run.update(dsp_attrs)
        run.update(arec.attributes)

        algo, train_time = self._train_algo(arec.algorithm, train)
        run['TrainTime'] = train_time

        preds, pred_time = self._predict(run_id, algo, test)
        run['PredTime'] = pred_time
        self._write_results('predictions', preds, run_id)

        recs, rec_time = self._recommend(run_id, algo, test, cand)
        run['RecTime'] = rec_time
        self._write_results('recommendations', recs, run_id)

        return run

    def _train_algo(self, algo, train):
        watch = util.Stopwatch()
        _logger.info('training algorithm %s on %d ratings', algo, len(train))
        # clone the algorithm in case some cannot be reused
        clone = util.clone(algo)
        clone.fit(train)
        watch.stop()
        _logger.info('trained algorithm %s in %s', algo, watch)
        return clone, watch.elapsed()

    def _predict(self, rid, algo, test):
        if not self.predict:
            return None, None
        if not isinstance(algo, Predictor):
            return None, None

        watch = util.Stopwatch()
        _logger.info('generating %d predictions for %s', len(test), algo)
        preds = predict(algo, test, nprocs=self.nprocs)
        watch.stop()
        _logger.info('generated predictions in %s', watch)
        preds['RunId'] = rid
        preds = preds[['RunId', 'user', 'item', 'rating', 'prediction']]
        return preds, watch.elapsed()

    def _recommend(self, rid, algo, test, candidates):
        if self.recommend is None:
            return None, None

        watch = util.Stopwatch()
        users = test.user.unique()
        _logger.info('generating recommendations for %d users for %s', len(users), algo)
        recs = recommend(algo, users, self.recommend, candidates, test,
                         nprocs=self.nprocs)
        watch.stop()
        _logger.info('generated recommendations in %s', watch)
        recs['RunId'] = rid
        return recs, watch.elapsed()

    def _write_run(self, run, run_data):
        if self.combine_output:
            run_df = pd.DataFrame(run_data)
            # overwrite files to show progress
            run_df.to_csv(self.run_csv, index=False)
            run_df.to_parquet(self.run_file, compression=None)
        else:
            rf = self.workdir / 'run-{}.json'.format(run['RunId'])
            with rf.open('w') as f:
                json.dump(run, f)

    def _write_results(self, name, df, run_id):
        if df is None:
            return

        if self.combine_output:
            out = self.workdir / '{}.parquet'.format(name)
            _logger.info('run %d: writing predictions to %s', run_id, out)
            append = run_id > 1
            util.write_parquet(out, df, append=append)
        else:
            out = self.workdir / '{}-{}.parquet'.format(name, run_id)
            _logger.info('run %d: writing predictions to %s', run_id, out)
            df.to_parquet(out)

    def collect_results(self):
        """
        Collect the results from non-combined runs into combined output files.
        """

        oc = self.combine_output
        try:
            self.combine_output = True
            n = self.run_count()
            runs = (self._read_json('run-{}.json', i+1) for i in range(n))
            runs = pd.DataFrame.from_records(runs)
            runs.to_parquet(self.run_file)
            runs.to_csv(self.run_csv, index=False)

            for i in range(n):
                preds = self._read_parquet('predictions-{}.parquet', i+1)
                self._write_results('predictions', preds, i+1)
                recs = self._read_parquet('recommendations-{}.parquet', i+1)
                self._write_results('recommendations', recs, i+1)
        finally:
            self.combine_output = oc

    def _read_parquet(self, name, *args):
        fn = self.workdir / name.format(*args)
        if not fn.exists():
            _logger.warning('file %s does not exist', fn)
            return None

        return pd.read_parquet(fn)

    def _read_json(self, name, *args):
        fn = self.workdir / name.format(*args)
        if not fn.exists():
            _logger.warning('file %s does not exist', fn)
            return {}

        with fn.open('r') as f:
            return json.load(f)
