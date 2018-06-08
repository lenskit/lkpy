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