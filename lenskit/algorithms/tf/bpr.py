import logging
import math

import pandas as pd
import numpy as np
from numba import njit

try:
    import tensorflow as tf
    import tensorflow.keras as k
except ImportError:
    tf = None

from lenskit import util
from lenskit.data import sparse_ratings
from .. import Predictor
from .util import init_tf_rng, check_tensorflow

_log = logging.getLogger(__name__)


@njit
def _sample_unweighted(mat):
    return np.random.randint(0, mat.ncols)


@njit
def _sample_weighted(mat):
    j = np.random.randint(0, mat.nnz)
    return mat.colinds[j]


@njit(nogil=True)
def _neg_sample(mat, uv, sample):
    """
    Sample the negative examples.  For each user in uv, it samples an item that
    they have not rated using rejection sampling.

    While this is embarassingly parallel, we do not parallelize because TensorFlow
    will request multiple batches in parallel.
    """
    n = len(uv)
    jv = np.empty(n, dtype=np.int32)
    sc = np.ones(n, dtype=np.int32)

    for i in range(n):
        u = uv[i]
        used = mat.row_cs(u)
        j = sample(mat)
        while np.any(used == j):
            j = sample(mat)
            sc[i] = sc[i] + 1
        jv[i] = j

    return jv, sc


if tf is not None:
    class BprLoss(k.losses.Loss):
        def call(self, y_true, y_pred):
            return k.backend.mean(-tf.math.log_sigmoid(y_pred))

    class BprInputs(k.utils.Sequence):
        def __init__(self, urmat, batch_size, neg_count, neg_weight, rng):
            super().__init__()
            self.n_items = urmat.ncols
            self.matrix = urmat
            self.users = urmat.rowinds()
            self.items = urmat.colinds
            self.rng = rng
            self.batch_size = batch_size
            self.neg_count = neg_count
            if neg_weight:
                self._sample = _sample_weighted
            else:
                self._sample = _sample_unweighted
            self.permutation = np.arange(self.matrix.nnz, dtype='i4')
            self.targets = np.ones(batch_size * neg_count)
            rng.shuffle(self.permutation)

        def __len__(self):
            return math.ceil(self.matrix.nnz / self.batch_size)

        def __getitem__(self, idx):
            _log.debug('preparing batch %d', idx)
            start = idx * self.batch_size
            end = min(start + self.batch_size, self.matrix.nnz)
            picked = self.permutation[start:end]
            if self.neg_count > 1:
                # expand picked size to sample more items
                picked = np.concatenate([picked for i in range(self.neg_count)])
            assert len(picked) == self.neg_count * (end - start)
            uv = self.users[picked]
            iv = self.items[picked]
            jv, j_samps = _neg_sample(self.matrix, uv, self._sample)
            assert all(jv < self.n_items)
            _log.debug('max sample count: %d', j_samps.max())
            return [uv.astype(np.int32),
                    iv.astype(np.int32),
                    jv.astype(np.int32)], self.targets[:len(picked)]

        def on_epoch_end(self):
            _log.info('re-shuffling')
            self.rng.shuffle(self.permutation)


class BPR(Predictor):
    """
    Bayesian Personalized Ranking with matrix factorization, optimized with TensorFlow.

    This is a basic TensorFlow implementation of the BPR algorithm _[BPR].

    .. _[BPR]:
        Rendle, S. et al. (2009) ‘BPR: Bayesian Personalized Ranking from Implicit
        Feedback’, in *Proceedings of the Twenty-Fifth Conference on Uncertainty in
        Artificial Intelligence*. AUAI Press (UAI ’09), pp. 452–461.

    User and item embedding matrices are regularized with :math:`L_2` regularization,
    governed by a regularization term :math:`\\lambda`.  Regularizations for the user
    and item embeddings are then computed as follows:

    .. math::
        \\lambda_u = \\lambda / |U| \\\\
        \\lambda_i = \\lambda / |I| \\\\

    This rescaling allows the regularization term to be independent of the number of
    users and items.

    Because the model is relatively simple, optimization works best with large
    batch sizes.

    Args:
        features(int): The number of latent features to learn.
        epochs(int): The number of epochs to train.
        batch_size(int):
            The Keras batch size.  This is the number of **positive** examples
            to sample in each batch.  If ``neg_count`` is greater than 1, the
            batch size will be similarly multipled.
        reg(double):
            The regularization term for the embedding vectors.
        neg_count(int):
            The number of negative examples to sample for each positive one.
        neg_weight(bool):
            Whether to weight negative sampling by popularity (``True``) or not.
        rng_spec:
            The random number generator initialization.

    Attributes:
        model: The Keras model.
    """

    model = None

    def __init__(self, features=50, *, epochs=5, batch_size=10000,
                 reg=0.02, neg_count=1, neg_weight=True, rng_spec=None):
        check_tensorflow()
        self.features = features
        self.epochs = epochs
        self.batch_size = batch_size
        self.reg = reg
        self.neg_count = neg_count
        self.neg_weight = neg_weight
        self.rng_spec = rng_spec

    def fit(self, ratings, **kwargs):
        timer = util.Stopwatch()
        rng = util.rng(self.rng_spec)

        matrix, users, items = sparse_ratings(ratings[['user', 'item']])

        _log.info('[%s] setting up model', timer)
        train, model = self._build_model(len(users), len(items))

        _log.info('[%s] preparing training dataset', timer)
        train_data = BprInputs(matrix, self.batch_size, self.neg_count, self.neg_weight, rng)

        _log.info('[%s] training model', timer)
        train.fit(train_data, epochs=self.epochs)

        _log.info('[%s] model finished', timer)

        self.user_index_ = users
        self.item_index_ = items
        self.model = model

        return self

    def _build_model(self, n_users, n_items):
        n_features = self.features
        _log.info('configuring TensorFlow model for %d features from %d users and %d items',
                  n_features, n_users, n_items)

        init_tf_rng(self.rng_spec)

        # User input layer
        u_input = k.Input(shape=(1,), dtype='int32', name='user')
        # User embedding layer.
        u_reg = k.regularizers.l2(self.reg / n_users)
        u_embed = k.layers.Embedding(input_dim=n_users, output_dim=n_features, input_length=1,
                                     embeddings_regularizer=u_reg,
                                     embeddings_initializer='random_normal',
                                     name='user-embed')
        # The embedding layer produces an extra dimension. Remove it.
        u_flat = k.layers.Flatten(name='user-vector')(u_embed(u_input))

        # Do the same thing for items
        i_input = k.Input(shape=(1,), dtype='int32', name='item')
        i_reg = k.regularizers.l2(self.reg / n_items)
        i_embed = k.layers.Embedding(input_dim=n_items, output_dim=n_features, input_length=1,
                                     embeddings_regularizer=i_reg,
                                     embeddings_initializer='random_normal',
                                     name='item-embed')
        i_flat = k.layers.Flatten(name='item-vector')(i_embed(i_input))

        # we need negative examples, run through the same embedding
        j_input = k.Input(shape=(1,), dtype='int32', name='neg-item')
        j_flat = k.layers.Flatten(name='neg-vector')(i_embed(j_input))

        # Score positive items with the dot product
        score = k.layers.Dot(name='pos-score', axes=1)([u_flat, i_flat])
        # And score negative items too
        neg_score = k.layers.Dot(name='neg-score', axes=1)([u_flat, j_flat])
        # Training is based on score differences
        train_score = k.layers.Subtract(name='score-diff')([score, neg_score])

        # Assemble the model for prediction
        model = k.Model([u_input, i_input], score, name='bpr-mf')
        # Assemble the training model and configure to optimize
        train = k.Model([u_input, i_input, j_input], train_score, name='bpr-train')
        train.compile('adam', BprLoss())

        return train, model

    def predict_for_user(self, user, items, ratings=None):
        if user not in self.user_index_:
            return pd.Series(np.nan, index=items)

        items = np.array(items)
        uno = self.user_index_.get_loc(user)
        inos = self.item_index_.get_indexer_for(items).astype('i4')
        good_inos = inos[inos >= 0]
        good_items = items[inos >= 0]
        unos = np.full(len(good_inos), uno, dtype='i4')
        _log.debug('scoring %d items for user %d', len(good_inos), user)

        ys = self.model([unos, good_inos], training=False)

        res = pd.Series(ys[:, 0], index=good_items)
        return res.reindex(items)

    def __getstate__(self):
        state = dict(self.__dict__)
        if self.model is not None:
            # we need to save the model
            del state['model']
            _log.info('extracting model config and weights')
            state['model_config'] = self.model.get_config()
            state['model_weights'] = self.model.get_weights()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if 'model_config' in state:
            _log.info('rehydrating model')
            self.model = k.Model.from_config(state['model_config'])
            self.model.set_weights(state['model_weights'])
            del self.model_config
            del self.model_weights
