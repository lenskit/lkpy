import logging

import pandas as pd
import numpy as np
try:
    import tensorflow as tf
    import tensorflow.keras as k
except ImportError:
    tf = None

from lenskit import util
from .. import Predictor
from .util import init_tf_rng, check_tensorflow

_log = logging.getLogger(__name__)


if tf is not None:
    class ScoreLayer(k.layers.Layer):
        """
        Custom layer for scoring.  We use this so that we can have the global mean value
        as an isolated variable for TensorFlow to optimize. Optimization is more effective
        if we initialize this variable to the global average rating.
        """
        def __init__(self, mean, **kwargs):
            super().__init__(**kwargs)
            self._mean = mean
            self.global_bias = tf.Variable(mean)

        def call(self, ub, ib, dot):
            return self.global_bias + ub + ib + dot

        def get_config(self):
            return {'mean': self._mean}


class IntegratedBiasMF(Predictor):
    """
    Biased matrix factorization model for explicit feedback, optimizing both bias
    and embeddings with TensorFlow.

    This is a basic TensorFlow implementation of the biased matrix factorization
    model for rating prediction:

    .. math::
        s(i|u) = b + b_u + b_i + \\vec{p}_u \\cdot \\vec{q_i}

    User and item embedding matrices are regularized with :math:`L_2` regularization,
    governed by a regularization term :math:`\\lambda`.  Regularizations for the user
    and item embeddings are then computed as follows:

    .. math::
        \\lambda_u = \\lambda / |U| \\\\
        \\lambda_i = \\lambda / |I| \\\\

    This rescaling allows the regularization term to be independent of the number of
    users and items.  The same rescaling applies to the bias regularization.

    Because the model is very simple, this algorithm works best with large
    batch sizes.

    This implementation uses TensorFlow to fit the entire model, including user/item
    biases and residuals, and uses TensorFlow to do the final predictions as well.
    Its code is suitable as an example of how to build a Keras/TensorFlow algorithm
    implementation for LensKit where TF used for the entire process.

    A variety of resources informed the design, most notably `this one`_ and
    `Chin-chi Hsu's example code`_.

    .. _this one: https://towardsdatascience.com/1fba34180699
    .. _Chin-chi Hsu's code: https://github.com/chinchi-hsu/KerasCollaborativeFiltering

    Args:
        features(int): The number of latent features to learn.
        epochs(int): The number of epochs to train.
        batch_size(int): The Keras batch size.
        reg(double):
            The regularization term for the embedding vectors.
        bias_reg(double):
            The regularization term for the bias vectors.
        rng_spec:
            The random number generator initialization.

    Attributes:
        model: The Keras model.
    """

    model = None

    def __init__(self, features=50, *, epochs=5, batch_size=10000,
                 reg=0.02, bias_reg=0.2, rng_spec=None):
        check_tensorflow()
        self.features = features
        self.epochs = epochs
        self.batch_size = batch_size
        self.reg = reg
        self.bias_reg = bias_reg
        self.rng_spec = rng_spec

    def fit(self, ratings, **kwargs):
        timer = util.Stopwatch()

        users = pd.Index(np.unique(ratings['user']))
        items = pd.Index(np.unique(ratings['item']))

        u_no = users.get_indexer(ratings['user'])
        i_no = items.get_indexer(ratings['item'])
        mean = np.mean(ratings['rating'].values, dtype='f4')  # TensorFlow is using 32-bits

        model = self._build_model(len(users), len(items), mean)

        _log.info('[%s] training model', timer)
        model.fit([u_no, i_no], ratings['rating'],
                  epochs=self.epochs, batch_size=self.batch_size)

        _log.info('[%s] model finished', timer)

        self.user_index_ = users
        self.item_index_ = items
        self.model = model

        return self

    def _build_model(self, n_users, n_items, mean):
        n_features = self.features
        _log.info('configuring TensorFlow model for %d features from %d users and %d items',
                  n_features, n_users, n_items)
        _log.info('global mean rating is %f', mean)

        init_tf_rng(self.rng_spec)

        # User input layer
        u_input = k.Input(shape=(1,), dtype='int32', name='user')
        # User embedding layer.
        u_reg = k.regularizers.l2(self.reg / n_users)
        u_embed = k.layers.Embedding(input_dim=n_users, output_dim=n_features, input_length=1,
                                     embeddings_regularizer=u_reg,
                                     embeddings_initializer='random_normal',
                                     name='user-embed')(u_input)
        # The embedding layer produces an extra dimension. Remove it.
        u_flat = k.layers.Flatten(name='user-vector')(u_embed)
        # User bias layer - it's a 1-dimensional embedding.
        ub_reg = k.regularizers.l2(self.bias_reg / n_users)
        u_bias = k.layers.Embedding(input_dim=n_users, output_dim=1, input_length=1,
                                    embeddings_regularizer=ub_reg,
                                    embeddings_initializer='random_normal',
                                    name='user-bias')(u_input)
        ub_flat = k.layers.Flatten(name='user-bv')(u_bias)

        # Do the same thing for items
        i_input = k.Input(shape=(1,), dtype='int32', name='item')
        i_reg = k.regularizers.l2(self.reg / n_items)
        i_embed = k.layers.Embedding(input_dim=n_items, output_dim=n_features, input_length=1,
                                     embeddings_regularizer=i_reg,
                                     embeddings_initializer='random_normal',
                                     name='item-embed')(i_input)
        i_flat = k.layers.Flatten(name='item-vector')(i_embed)
        ib_reg = k.regularizers.l2(self.bias_reg / n_items)
        i_bias = k.layers.Embedding(input_dim=n_items, output_dim=1, input_length=1,
                                    embeddings_regularizer=ib_reg,
                                    embeddings_initializer='random_normal',
                                    name='item-bias')(i_input)
        ib_flat = k.layers.Flatten(name='item-bv')(i_bias)

        # Predict ratings using a dot product of users and items
        dot = k.layers.Dot(name='resid-score', axes=1)([u_flat, i_flat])
        # score = k.layers.Add(name='score')([u_bias, i_bias, dot])
        score = ScoreLayer(mean, name='score')(ub_flat, ib_flat, dot)

        # Assemble the model and configure to optimize
        model = k.Model([u_input, i_input], score, name='int-mf')
        model.compile('adam', 'mean_squared_error')

        return model

    def predict_for_user(self, user, items, ratings=None):
        if user not in self.user_index_:
            return pd.Series(np.nan, index=items)

        items = np.array(items)
        uno = self.user_index_.get_loc(user)
        inos = self.item_index_.get_indexer_for(items).astype('i4')
        good_inos = inos[inos >= 0]
        good_items = items[inos >= 0]
        unos = np.full(len(good_inos), uno, dtype='i4')
        _log.debug('predicting %d items for user %d', len(good_inos), user)

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
            self.model = k.Model.from_config(state['model_config'], custom_objects={
                'ScoreLayer': ScoreLayer
            })
            self.model.set_weights(state['model_weights'])
