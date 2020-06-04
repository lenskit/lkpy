import logging

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as k

from lenskit import util
from ..mf_common import BiasMFPredictor
from ..basic import Bias

_log = logging.getLogger(__name__)


class BiasedMF(BiasMFPredictor):
    """
    Biased matrix factorization model for explicit feedback, optimized with
    TensorFlow.

    This is a basic TensorFlow implementation of the biased matrix factorization
    model for rating prediction:

    .. math::
        s(i|u) = b + b_u + b_i + \\vec{p}_u \\cdot \\vec{q_i}

    Because the model is very simple, this algorithm works best with large
    batch sizes.

    This implementation uses :class:`lenskit.algorithms.basic.Bias` for computing
    the biases, and uses TensorFlow to fit a matrix factorization on the residuals.
    It then extracts the resulting matrices, and relies on :class:`BiasedMFPredictor`
    to implement the prediction logic, like :class:`lenskit.algorithms.als.BiasedMF`.
    Its code is suitable as an example of how to build a Keras/TensorFlow algorithm
    implementation for LensKit where TF is only used in the train stage.

    A variety of resources informed the design, most notably `this one`_.

    .. _this one:: https://towardsdatascience.com/building-a-book-recommendation-system-using-keras-1fba34180699
    """

    def __init__(self, features=50, *, bias=True, damping=5, epochs=5, batch_size=10000, reg=0.02, rng_spec=None):
        self.features = features
        self.epochs = epochs
        self.batch_size = batch_size
        self.reg = reg
        if bias is True:
            self.bias = Bias(damping)
        else:
            self.bias = bias
        self.rng_spec = None

    def fit(self, ratings, **kwargs):
        timer = util.Stopwatch()
        normed = self.bias.fit_transform(ratings, indexes=True)
        graph, model = self._build_model(len(self.bias.user_offsets_),
                                         len(self.bias.item_offsets_))

        _log.info('[%s] training model', timer)
        with graph.as_default():
            model.fit([normed['uidx'], normed['iidx']], normed['rating'],
                      epochs=self.epochs, batch_size=self.batch_size)

            _log.info('[%s] model finished, extracting weights')
            self.user_features_ = model.get_layer('user-embed').get_weights()[0]
            self.item_features_ = model.get_layer('item-embed').get_weights()[0]

        self.global_bias_ = self.bias.mean_
        self.user_bias_ = self.bias.user_offsets_.values
        self.item_bias_ = self.bias.item_offsets_.values
        self.user_index_ = self.bias.user_index
        self.item_index_ = self.bias.item_index

        return self

    def _build_model(self, n_users, n_items):
        n_features = self.features
        _log.info('configuring TensorFlow model for %d features from %d users and %d items',
                  n_features, n_users, n_items)
        rng = util.rng(self.rng_spec)
        graph = tf.Graph()
        graph.seed = rng.integers(2**32)
        _log.info('using random seed %s', graph.seed)
        with graph.as_default():
            # User input layer
            u_input = k.Input(shape=(1,), dtype='int32', name='user')
            # User embedding layer. We regularize the output, not embedding, so we don't have
            # to scale the regularization term with the number of users.
            u_reg = k.regularizers.l2(self.reg)
            u_embed = k.layers.Embedding(input_dim=n_users, output_dim=n_features, input_length=1,
                                         activity_regularizer=u_reg,
                                         embeddings_initializer='random_normal',
                                         name='user-embed')(u_input)
            # The embedding layer produces an extra dimension. Remove it.
            u_flat = k.layers.Flatten(name='user-vector')(u_embed)

            # Do the same thing for items
            i_input = k.Input(shape=(1,), dtype='int32', name='item')
            i_reg = k.regularizers.l2(self.reg)
            i_embed = k.layers.Embedding(input_dim=n_items, output_dim=n_features, input_length=1,
                                         activity_regularizer=i_reg,
                                         embeddings_initializer='random_normal',
                                         name='item-embed')(i_input)
            i_flat = k.layers.Flatten(name='item-vector')(i_embed)

            # Predict ratings using a dot product of users and items
            prod = k.layers.Dot(name='score', axes=1)([u_flat, i_flat])

            # Assemble the model and configure to optimize
            model = k.Model([u_input, i_input], prod, name='classic-mf')
            model.compile('adam', 'mean_squared_error')

        return graph, model

    def predict_for_user(self, user, items, ratings=None):
        # look up user index
        return self.score_by_ids(user, items)
