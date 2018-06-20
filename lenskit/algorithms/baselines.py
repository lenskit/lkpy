"""
Non-personalized or lightly-personalized baseline algorithms for recommendation
and rating prediction.
"""

from collections import namedtuple
import logging

import pandas as pd

from .. import util as lku
from .. import check

_logger = logging.getLogger(__package__)

BiasModel = namedtuple('BiasModel', ['mean', 'items', 'users'])
BiasModel.__doc__ = "Trained model for the :py:class:`Bias` algorithm."


class Bias:
    """
    A user-item bias rating prediction algorithm.  This implements the following
    predictor algorithm:

    .. math::
       s(u,i) = \mu + b_i + b_u

    where :math:`\mu` is the global mean rating, :math:`b_i` is item bias, and
    :math:`b_u` is the user bias.  With the provided damping values
    :math:`\\beta_{\\mathrm{u}}` and :math:`\\beta_{\\mathrm{i}}`, they are computed
    as follows:

    .. math::
       \\begin{align*}
       \\mu & = \\frac{\\sum_{r_{ui} \\in R} r_{ui}}{|R|} &
       b_i & = \\frac{\\sum_{r_{ui} \\in R_i} (r_{ui} - \\mu)}{|R_i| + \\beta_{\\mathrm{i}}} &
       b_u & = \\frac{\\sum_{r_{ui} \\in R_u} (r_{ui} - \\mu - b_i)}{|R_u| + \\beta_{\\mathrm{u}}}
       \\end{align*}

    The damping values can be interpreted as the number of default (mean) ratings to assume
    *a priori* for each user or item, damping low-information users and items towards a mean instead
    of permitting them to take on extreme values based on few ratings.

    Args:
        items: whether to compute item biases
        users: whether to compute user biases
        damping(number or tuple):
            Bayesian damping to apply to computed biases.  Either a number, to
            damp both user and item biases the same amount, or a (user,item) tuple
            providing separate damping values.
    """

    def __init__(self, items=True, users=True, damping=0.0):
        self._include_items = items
        self._include_users = users
        if isinstance(damping, tuple):
            self.user_damping, self.item_damping = damping
        else:
            self.user_damping = damping
            self.item_damping = damping

        check.check_value(self.user_damping >= 0, "user damping value {} must be nonnegative",
                          self.user_damping)
        check.check_value(self.item_damping >= 0, "item damping value {} must be nonnegative",
                          self.item_damping)

    def train(self, data):
        """
        Train the bias model on some rating data.

        Args:
            data (DataFrame): a data frame of ratings. Must have at least `user`,
                              `item`, and `rating` columns.

        Returns:
            BiasModel: a trained model with the desired biases computed.
        """

        _logger.info('building bias model for %d ratings', len(data))
        mean = data.rating.mean()
        mean = lku.compute(mean)
        _logger.info('global mean: %.3f', mean)
        nrates = data.assign(rating=lambda df: df.rating - mean)

        if self._include_items:
            group = nrates.groupby('item').rating
            item_offsets = self._mean(group, self.item_damping)
            item_offsets = lku.compute(item_offsets)
            _logger.info('computed means for %d items', len(item_offsets))
        else:
            item_offsets = None

        if self._include_users:
            if item_offsets is not None:
                nrates = nrates.join(pd.DataFrame(item_offsets), on='item', how='inner',
                                     rsuffix='_im')
                nrates = nrates.assign(rating=lambda df: df.rating - df.rating_im)

            user_offsets = self._mean(nrates.groupby('user').rating, self.user_damping)
            user_offsets = lku.compute(user_offsets)
            _logger.info('computed means for %d users', len(user_offsets))
        else:
            user_offsets = None

        return BiasModel(mean, item_offsets, user_offsets)

    def predict(self, model, user, items, ratings=None):
        """
        Compute predictions for a user and items.  Unknown users and items
        are assumed to have zero bias.

        Args:
            model (BiasModel): the trained model to use.
            user: the user ID
            items (array-like): the items to predict
            ratings (pandas.Series): the user's ratings (indexed by item id); if
                                 provided, will be used to recompute the user's
                                 bias at prediction time.

        Returns:
            pandas.Series: scores for the items, indexed by item id.
        """

        idx = pd.Index(items)
        preds = pd.Series(model.mean, idx)

        if model.items is not None:
            preds = preds + model.items.reindex(items, fill_value=0)

        if self._include_users and ratings is not None:
            uoff = ratings - model.mean
            if model.items is not None:
                uoff = uoff - model.items
            umean = uoff.mean()
            preds = preds + umean
        elif model.users is not None:
            umean = model.users.get(user, 0.0)
            _logger.debug('using mean(user %s) = %.3f', user, umean)
            preds = preds + umean

        return preds

    def _mean(self, series, damping):
        if damping is not None and damping > 0:
            return series.sum() / (series.count() + damping)
        else:
            return series.mean()
