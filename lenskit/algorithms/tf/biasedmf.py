import logging

try:
    import tensorflow as tf
    import tensorflow.keras as k
except ImportError:
    tf = None

from lenskit import util
from ..mf_common import MFPredictor
from ..bias import Bias
from .util import init_tf_rng, check_tensorflow

_log = logging.getLogger(__name__)


class BiasedMF(MFPredictor):
    """
    Biased matrix factorization model for explicit feedback, optimized with
    TensorFlow.

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
    users and items.

    Because the model is very simple, this algorithm works best with large
    batch sizes.

    This implementation uses :class:`lenskit.algorithms.bias.Bias` for computing
    the biases, and uses TensorFlow to fit a matrix factorization on the residuals.
    It then extracts the resulting matrices, and relies on :class:`MFPredictor`
    to implement the prediction logic, like :class:`lenskit.algorithms.als.BiasedMF`.
    Its code is suitable as an example of how to build a Keras/TensorFlow algorithm
    implementation for LensKit where TF is only used in the train stage.

    A variety of resources informed the design, most notably `this one`_.

    .. _this one: https://towardsdatascience.com/1fba34180699

    Args:
        features(int): The number of latent features to learn.
        bias: The bias model to use.
        damping: The bias damping, if ``bias`` is ``True``.
        epochs(int): The number of epochs to train.
        batch_size(int): The Keras batch size.
        reg(double):
            The regularization term :math:`\\lambda` used to derive embedding vector
            regularizations.
        rng_spec:
            The random number generator initialization.
    """

    def __init__(self, features=50, *, bias=True, damping=5,
                 epochs=5, batch_size=10000, reg=0.02, rng_spec=None):
        check_tensorflow()
        self.features = features
        self.epochs = epochs
        self.batch_size = batch_size
        self.reg = reg
        if bias is True:
            self.bias = Bias(damping)
        else:
            self.bias = bias
        self.rng_spec = rng_spec

    def fit(self, ratings, **kwargs):
        timer = util.Stopwatch()
        normed = self.bias.fit_transform(ratings, indexes=True)
        model = self._build_model(len(self.bias.user_offsets_),
                                  len(self.bias.item_offsets_))

        _log.info('[%s] training model', timer)
        model.fit([normed['uidx'], normed['iidx']], normed['rating'],
                  epochs=self.epochs, batch_size=self.batch_size)

        _log.info('[%s] model finished, extracting weights', timer)
        self.user_features_ = model.get_layer('user-embed').get_weights()[0]
        self.item_features_ = model.get_layer('item-embed').get_weights()[0]

        self.user_index_ = self.bias.user_index
        self.item_index_ = self.bias.item_index

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
                                     name='user-embed')(u_input)
        # The embedding layer produces an extra dimension. Remove it.
        u_flat = k.layers.Flatten(name='user-vector')(u_embed)

        # Do the same thing for items
        i_input = k.Input(shape=(1,), dtype='int32', name='item')
        i_reg = k.regularizers.l2(self.reg / n_items)
        i_embed = k.layers.Embedding(input_dim=n_items, output_dim=n_features, input_length=1,
                                     embeddings_regularizer=i_reg,
                                     embeddings_initializer='random_normal',
                                     name='item-embed')(i_input)
        i_flat = k.layers.Flatten(name='item-vector')(i_embed)

        # Predict ratings using a dot product of users and items
        prod = k.layers.Dot(name='score', axes=1)([u_flat, i_flat])

        # Assemble the model and configure to optimize
        model = k.Model([u_input, i_input], prod, name='classic-mf')
        model.compile('adam', 'mean_squared_error')

        return model

    def predict_for_user(self, user, items, ratings=None):
        # look up user index
        preds = self.score_by_ids(user, items)
        preds = self.bias.inverse_transform_user(user, preds)
        return preds
