import logging
from copy import copy

import numpy as np
from numba import njit, prange

from . import basic
from . import Predictor, Trainable
from .mf_common import BiasMFModel, MFModel
from ..matrix import sparse_ratings, CSR
from .. import util
from ..math.solve import _dposv

_logger = logging.getLogger(__name__)


@njit(parallel=True, nogil=True)
def _train_matrix(mat: CSR, other: np.ndarray, reg: float):
    "One half of an explicit ALS training round."
    nr = mat.nrows
    nf = other.shape[1]
    regI = np.identity(nf) * reg
    assert mat.ncols == other.shape[0]
    result = np.zeros((nr, nf))
    for i in prange(nr):
        cols = mat.row_cs(i)
        if len(cols) == 0:
            continue

        vals = mat.row_vs(i)
        M = other[cols, :]
        MMT = M.T @ M
        # assert MMT.shape[0] == ctx.n_features
        # assert MMT.shape[1] == ctx.n_features
        A = MMT + regI * len(cols)
        V = M.T @ vals
        # and solve
        _dposv(A, V, True)
        result[i, :] = V

    return result


@njit(parallel=True, nogil=True)
def _train_implicit_matrix(mat: CSR, other: np.ndarray, reg: float):
    "One half of an implicit ALS training round."
    nr = mat.nrows
    nc = other.shape[0]
    nf = other.shape[1]
    assert mat.ncols == nc
    regmat = np.identity(nf) * reg
    Ot = other.T
    OtO = Ot @ other
    OtOr = OtO + regmat
    assert OtO.shape[0] == OtO.shape[1]
    assert OtO.shape[0] == nf
    result = np.zeros((nr, nf))
    for i in prange(nr):
        cols = mat.row_cs(i)
        if len(cols) == 0:
            continue

        rates = mat.row_vs(i)

        # we can optimize by only considering the nonzero entries of Cu-I
        # this means we only need the corresponding matrix columns
        M = other[cols, :]
        # Compute M^T C_u M, restricted to these nonzero entries
        MMT = (M.T.copy() * rates) @ M
        # assert MMT.shape[0] == ctx.n_features
        # assert MMT.shape[1] == ctx.n_features
        # Build the matrix for solving
        A = OtOr + MMT
        # Compute RHS - only used columns (p_ui != 0) values needed
        # Cu is rates + 1 for the cols, so just trim Ot
        y = Ot[:, cols] @ (rates + 1.0)
        # and solve
        _dposv(A, y, True)
        # assert len(uv) == ctx.n_features
        result[i, :] = y

    return result


class BiasedMF(Predictor, Trainable):
    """
    Biased matrix factorization trained with alternating least squares [ZWSP2008]_.  This is a
    prediction-oriented algorithm suitable for explicit feedback data.

    .. [ZWSP2008] Yunhong Zhou, Dennis Wilkinson, Robert Schreiber, and Rong Pan. 2008.
        Large-Scale Parallel Collaborative Filtering for the Netflix Prize.
        In +Algorithmic Aspects in Information and Management_, LNCS 5034, 337–348.
        DOI `10.1007/978-3-540-68880-8_32 <http://dx.doi.org/10.1007/978-3-540-68880-8_32>`_.

    Args:
        features(int): the number of features to train
        iterations(int): the number of iterations to train
        reg(double): the regularization factor
        damping(double): damping factor for the underlying mean

    Attributes:
        features(int): the number of features.
        iterations(int): the number of training iterations.
        regularization(double): the regularization factor.
        damping(double): the mean damping.
    """
    timer = None

    def __init__(self, features, iterations=20, reg=0.1, damping=5):
        self.features = features
        self.iterations = iterations
        self.regularization = reg
        self.damping = damping

    def train(self, ratings, bias=None):
        """
        Run ALS to train a model.

        Args:
            ratings: the ratings data frame.
            bias(.bias.BiasModel): a pre-trained bias model to use.

        Returns:
            BiasMFModel: The trained biased MF model.
        """
        self.timer = util.Stopwatch()

        current, uctx, ictx = self._initial_model(ratings, bias)

        _logger.info('[%s] training biased MF model with ALS for %d features',
                     self.timer, self.features)
        for epoch, model in enumerate(self._train_iters(current, uctx, ictx)):
            current = model

        _logger.info('trained model in %s', self.timer)

        return current

    def _initial_model(self, ratings, bias=None):
        "Initialize a model and build contexts."
        bias = self._get_bias(bias, ratings)
        rmat, users, items = sparse_ratings(ratings)
        n_users = len(users)
        n_items = len(items)

        rmat, bias = self._normalize(rmat, users, items, bias)
        assert len(bias.users) == n_users and len(bias.items) == n_items

        _logger.debug('setting up contexts')
        trmat = rmat.transpose()

        _logger.debug('initializing item matrix')
        imat = np.random.randn(n_items, self.features) * 0.01
        umat = np.full((n_users, self.features), np.nan)

        return BiasMFModel(users, items, bias, umat, imat), rmat, trmat

    def _get_bias(self, bias, ratings):
        "Ensure we have a suitable set of bias terms for the model."
        if bias is None:
            _logger.info('[%s] training bias model', self.timer)
            bias = basic.Bias(damping=self.damping).train(ratings)
        # unpack the bias
        if isinstance(bias, basic.BiasModel):
            return bias
        else:
            # we have a single global bias
            return basic.BiasModel(bias, None, None)

    def _normalize(self, ratings, users, items, bias):
        "Apply bias normalization to the data in preparation for training."
        n_users = len(users)
        n_items = len(items)
        _logger.info('shape %dx%d, u %d, i %d', ratings.nrows, ratings.ncols, n_users, n_items)
        assert ratings.nrows == n_users
        assert ratings.ncols == n_items

        _logger.info('[%s] normalizing %dx%d matrix (%d nnz)',
                     self.timer, n_users, n_items, ratings.nnz)
        ratings.values = ratings.values - bias.mean
        ibias = bias.items
        if ibias is not None:
            ibias = ibias.reindex(items, fill_value=0)
            ratings.values = ratings.values - ibias.values[ratings.colinds]
        ubias = bias.users
        if ubias is not None:
            ubias = ubias.reindex(users, fill_value=0)
            # create a user index array the size of the data
            reps = np.repeat(np.arange(len(users), dtype=np.int32),
                             ratings.row_nnzs())
            assert len(reps) == ratings.nnz
            # subtract user means
            ratings.values = ratings.values - ubias.values[reps]
            del reps

        return ratings, basic.BiasModel(bias.mean, ibias, ubias)

    def _train_iters(self, current, uctx, ictx):
        "Generator of training iterations."
        for epoch in range(self.iterations):
            umat = _train_matrix(uctx, current.item_features, self.regularization)
            _logger.debug('[%s] finished user epoch %d', self.timer, epoch)
            imat = _train_matrix(ictx, umat, self.regularization)
            _logger.debug('[%s] finished item epoch %d', self.timer, epoch)
            di = np.linalg.norm(imat - current.item_features, 'fro')
            du = np.linalg.norm(umat - current.user_features, 'fro')
            _logger.info('[%s] finished epoch %d (|ΔI|=%.3f, |ΔU|=%.3f)', self.timer, epoch, di, du)
            current = copy(current)
            current.user_features = umat
            current.item_features = imat
            yield current

    def predict(self, model: BiasMFModel, user, items, ratings=None):
        # look up user index
        return model.score_by_ids(user, items)

    def save_model(self, model, path):
        model.save(path)

    def load_model(self, path):
        return BiasMFModel.load(path)

    def __str__(self):
        return 'als.BiasedMF(features={}, regularization={})'.\
            format(self.features, self.regularization)


class ImplicitMF(Predictor, Trainable):
    """
    Implicit matrix factorization trained with alternating least squares [HKV2008]_.  This
    algorithm outputs 'predictions', but they are not on a meaningful scale.  If its input
    data contains ``rating`` values, these will be used as the 'confidence' values; otherwise,
    confidence will be 1 for every rated item.

    .. [HKV2008] Y. Hu, Y. Koren, and C. Volinsky. 2008.
       Collaborative Filtering for Implicit Feedback Datasets.
       In _Proceedings of the 2008 Eighth IEEE International Conference on Data Mining_, 263–272.
       DOI `10.1109/ICDM.2008.22 <http://dx.doi.org/10.1109/ICDM.2008.22>`_

    Args:
        features(int): the number of features to train
        iterations(int): the number of iterations to train
        reg(double): the regularization factor
        weight(double): the scaling weight for positive samples (:math:`\\alpha` in [HKV2008]_).
    """
    timer = None

    def __init__(self, features, iterations=20, reg=0.1, weight=40):
        self.features = features
        self.iterations = iterations
        self.regularization = reg
        self.weight = weight

    def train(self, ratings):
        self.timer = util.Stopwatch()
        current, uctx, ictx = self._initial_model(ratings)

        _logger.info('[%s] training implicit MF model with ALS for %d features',
                     self.timer, self.features)
        _logger.info('have %d observations for %d users and %d items',
                     uctx.nnz, uctx.nrows, ictx.nrows)
        for model in self._train_iters(current, uctx, ictx):
            current = model

        _logger.info('[%s] finished training model with %d features',
                     self.timer, current.n_features)

        return current

    def _train_iters(self, current, uctx, ictx):
        "Generator of training iterations."
        for epoch in range(self.iterations):
            umat = _train_implicit_matrix(uctx, current.item_features,
                                          self.regularization)
            _logger.debug('[%s] finished user epoch %d', self.timer, epoch)
            imat = _train_implicit_matrix(ictx, umat, self.regularization)
            _logger.debug('[%s] finished item epoch %d', self.timer, epoch)
            di = np.linalg.norm(imat - current.item_features, 'fro')
            du = np.linalg.norm(umat - current.user_features, 'fro')
            _logger.info('[%s] finished epoch %d (|ΔI|=%.3f, |ΔU|=%.3f)', self.timer, epoch, di, du)
            current = copy(current)
            current.user_features = umat
            current.item_features = imat
            yield current

    def _initial_model(self, ratings):
        "Initialize a model and build contexts."

        rmat, users, items = sparse_ratings(ratings)
        n_users = len(users)
        n_items = len(items)

        _logger.debug('setting up contexts')
        # force values to exist
        if rmat.values is None:
            rmat.values = np.ones(rmat.nnz)
        rmat.values *= self.weight
        trmat = rmat.transpose()

        imat = np.random.randn(n_items, self.features) * 0.01
        imat = np.square(imat)
        umat = np.full((n_users, self.features), np.nan)

        return MFModel(users, items, umat, imat), rmat, trmat

    def predict(self, model: MFModel, user, items, ratings=None):
        # look up user index
        return model.score_by_ids(user, items)

    def save_model(self, model, path):
        model.save(path)

    def load_model(self, path):
        return MFModel.load(path)

    def __str__(self):
        return 'als.ImplicitMF(features={}, regularization={}, w={})'.\
            format(self.features, self.regularization, self.weight)
