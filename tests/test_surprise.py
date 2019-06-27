# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:42:44 2019

@author: mohamed.cherif
"""

import logging
import pickle

import pandas as pd
import numpy as np

from pytest import mark

try:
    import surprise
    have_surprise = True
except ImportError:
    have_surprise = False

import lenskit.util.test as lktu
from lenskit.algorithms.surprise import (SurpriseKNNBasic, SurpriseCoClustering,
 SurpriseKNNBaseline, SurpriseKNNWithMeans, SurpriseKNNWithZScore, SurpriseNMF,
 SurpriseNormalPredictor, SurpriseSlopeOne, SurpriseSVD, SurpriseSVDpp)

from lenskit import util

_log = logging.getLogger(__name__)

simple_df = pd.DataFrame({'item': [1, 1, 2, 3],
                          'user': [10, 12, 10, 13],
                          'rating': [4.0, 3.0, 5.0, 2.0]})

surprise_algo_list = [SurpriseKNNBasic, SurpriseCoClustering,
 SurpriseKNNBaseline, SurpriseKNNWithMeans, SurpriseKNNWithZScore, SurpriseNMF,
 SurpriseNormalPredictor, SurpriseSlopeOne, SurpriseSVD, SurpriseSVDpp]
# none knn based algos and none normal predictor list for test
surprise_algo_list_none_knn_based = [SurpriseCoClustering, SurpriseNMF
                                     , SurpriseSlopeOne, SurpriseSVD, SurpriseSVDpp]

@mark.slow
@mark.skipif(not have_surprise, reason='surprise not installed')
def test_surprise_train_rec():
    """
    recommendation test over Surprise algorithms
    """
    ratings = lktu.ml_test.ratings
    for algo in surprise_algo_list_none_knn_based:
        algo = algo()    
        ret = algo.fit(ratings= ratings)
        assert ret is algo
    
        recs = algo.recommend(100, n=20)
        assert len(recs) == 20
    
        _log.info('serializing surprise model')
        mod = pickle.dumps(algo)
        _log.info('serialized to %d bytes')
        a2 = pickle.loads(mod)
    
        r2 = a2.recommend(100, n=20)
        assert len(r2) == 20
        assert all(r2 == recs)


@mark.slow
@mark.eval
@mark.skipif(not have_surprise, reason='surprise not installed')
@mark.skipif(not lktu.ml100k.available, reason='ML100K data not present')
def test_surprise_batch_accuracy():
    import lenskit.crossfold as xf
    from lenskit import batch, topn

    ratings = lktu.ml100k.ratings
    
    def eval(train, test, algo_t):
        _log.info('running training')
        train['rating'] = train.rating.astype(np.float_)
        algo = util.clone(algo_t)
        algo.fit(train)
        users = test.user.unique()
        _log.info('testing %d users', len(users))
        candidates = topn.UnratedCandidates(train)
        recs = batch.recommend(algo, users, 100, candidates)
        return recs

    # Running evaluation for none based knn algorithms 
    for algo_t in surprise_algo_list_none_knn_based:
            
        folds = list(xf.partition_users(ratings, 5, xf.SampleFrac(0.2)))
        test = pd.concat(f.test for f in folds)
    
        recs = pd.concat(eval(train, test, algo_t()) for (train, test) in folds)
    
        _log.info('analyzing recommendations')
        rla = topn.RecListAnalysis()
        rla.add_metric(topn.ndcg)
        results = rla.compute(recs, test)
        dcg = results.ndcg
        _log.info('nDCG for %d users is %.4f', len(dcg), dcg.mean())
        assert dcg.mean() > 0



@mark.skipif(not have_surprise, reason='surprise not installed')
def test_surprise_pickle_untrained(tmp_path):
    mf = tmp_path / 'bpr.dat'
    for algorithm_class in surprise_algo_list:
        algo = algorithm_class()    
        with open(mf,'wb') as f:
            pickle.dump(algo, f)
    
        with open(mf,'rb') as f:
            a2 = pickle.load(f)
    
        assert a2 is not algo
        

def compare_params_to_dict(class_name, params):
    """
    function used to compare between params and algo instance params
    """
    # Instantiate algo with params
    algo = class_name(**params)
    # get instantiated algo params
    algo_params = algo.get_params()
    # Assertion loop through params
    for param_key in params.keys():
        assert params.get(param_key) == algo_params.get(param_key) 


        
def test_surprise_KNNBasic_params():
    params = {"k": 40, "min_k": 2, "sim_options": {'user_based': False}}
    compare_params_to_dict(SurpriseKNNBasic, params)



def test_surprise_KNNWithMeans_params():
    params = {"k": 10, "min_k": 1, "sim_options": {'user_based': True}}
    compare_params_to_dict(SurpriseKNNWithMeans, params)



def test_surprise_KNNWithZScore_params():
    params = {"k": 30, "min_k": 1, "sim_options": {'user_based': True, 'name': 'cosine'}}
    compare_params_to_dict(SurpriseKNNWithZScore, params)
    
    
def test_surprise_KNNBaseline_params():
    params = {"k": 20, "min_k": 1, "sim_options": {'user_based': False, 'name': 'cosine'}}
    compare_params_to_dict(SurpriseKNNBaseline, params)    


def test_surprise_CoClustering_params():
    params = {"n_cltr_u": 4, "n_cltr_i": 10, "n_epochs": 30}
    compare_params_to_dict(SurpriseCoClustering, params) 



def test_surprise_SVD_params():
    params = {"n_factors": 50, "biased": False, "n_epochs": 30, "lr_qi":0.003}
    compare_params_to_dict(SurpriseSVD, params) 


def test_surprise_SVDpp_params():
    params = {"n_factors": 80, "n_epochs": 50, "lr_bi":0.006}
    compare_params_to_dict(SurpriseSVDpp, params) 


def test_surprise_NMF_params():
    params = {"n_factors": 10, "biased": True, "n_epochs": 70, "lr_bu":0.006}
    compare_params_to_dict(SurpriseNMF, params) 






