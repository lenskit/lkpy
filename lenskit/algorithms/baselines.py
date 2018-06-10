import sys
from collections import namedtuple
import logging

import pandas as pd
import numpy as np

_logger = logging.getLogger(__package__)

class Bias:
	"""
	A rating-bias rating prediction algorithm.
	"""

	Model = namedtuple('BiasModel', ['mean', 'items', 'users'])

	def __init__(self, items=True, users=True):
		self._include_items = items
		self._include_users = users

	def train(self, data: pd.DataFrame) -> Model:
		"""
		Train the bias model on some rating data.
		"""
		
		_logger.info('building bias model for %d ratings', len(data))
		mean = data.rating.mean()
		_logger.info('global mean: %.3f', mean)

		# index for next step
		irates = data.set_index(['user', 'item']).rating - mean
		
		if self._include_items:
			item_offsets = irates.groupby('item').mean()
			_logger.info('computed means for %d items', len(item_offsets))
		else:
			item_offsets = None

		if self._include_users:
			if item_offsets is not None:
				irates = irates - item_offsets
			user_offsets = irates.groupby('user').mean()
			_logger.info('computed means for %d users', len(user_offsets))
		else:
			user_offsets = None

		return self.Model(mean, item_offsets, user_offsets)

	def predict(self, model: Model, user, items, ratings=None) -> pd.DataFrame:
		"""
		Compute predictions for a user and items.
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