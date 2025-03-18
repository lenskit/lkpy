import logging
import pickle

import numpy as np
import pandas as pd
import torch

from pytest import approx, mark

from lenskit.data import Dataset, ItemList, RecQuery, from_interactions_df, load_movielens_df
from lenskit.flexmf import FlexMFExplicitScorer
from lenskit.metrics import quick_measure_model
from lenskit.testing import BasicComponentTests, ScorerTests, wantjit


class TestExplicitALS(BasicComponentTests, ScorerTests):
    component = FlexMFExplicitScorer
