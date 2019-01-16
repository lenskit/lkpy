import pandas as pd
import numpy as np
from lenskit import batch, topn, util
from lenskit import crossfold as xf
from lenskit.algorithms import als
from lenskit.algorithms import Recommender

# read in the movielens 100k ratings with pandas
# https://grouplens.org/datasets/movielens/100k/
ratings = pd.read_csv('ml-100k/u.data', sep='\t',
        names=['user', 'item', 'rating', 'timestamp'])

# define the algorithm we will use
# In this case we use an alternating least square
# implementation of matrix factorization
# We train 6 features
# https://lkpy.lenskit.org/en/stable/mf.html#module-lenskit.algorithms.als
algoAls = als.BiasedMF(6)

# Clone the algoritm as otherwise some
# algorithms can behave strange after they
# fitted multiple times
fittableALS = util.clone(algoAls)

# split the data in a test and a training set
# for each user leave one row out for test purpose
data = ratings
nb_partitions = 1
splits = xf.partition_users(data, nb_partitions, xf.SampleN(1))
for (trainSet, testSet) in splits:
    train = trainSet
    test = testSet

# Build a model
modelAls = fittableALS.fit(train)

# Inspect the user-feature matrix (numpy array)
print(modelAls.user_features_[0:10])
print(modelAls.user_features_.size)

# Inspect the item-feature matrix (numpy array)
print(modelAls.item_features_[0:10])
print(modelAls.item_features_.size)

# Get all the users in the test set (numpy array)
users = test.user.unique()
print(users.size)


# See that the users are ranked differently in the matrix
print(modelAls.user_index_[0:10])
first = modelAls.user_index_[0]
firstUser = np.array([first], np.int64)

# Get recommendation for a user
# As als.BiasedMF is not implementing Recommend
# We need to first adapt the model
# https://lkpy.lenskit.org/en/stable/interfaces.html#recommendation
rec = Recommender.adapt(modelAls)
# recs = rec.recommend(firstUser, 10) #Gives error

# Get 10 recommendations for a user (pandas dataframe)
recs = batch.recommend(modelAls, firstUser,
                       10, topn.UnratedCandidates(train), test)
print(recs)

# Get the first recommended item
firstRec = recs.iloc[0, 0]
firstRecScore = recs.iloc[0, 1]
print(firstRec)

# Get the explanation of the recommendation
# Get the index of the items
items = modelAls.item_index_
# Find the index of the first item
indexFirstRec = items.get_loc(firstRec)
# Get the feature values of the user
userProfile = modelAls.user_features_[0]
itemProfile = modelAls.item_features_[indexFirstRec]
print(userProfile)
print(itemProfile)
# Get the MF score of item and user
mfScore = np.dot(userProfile, itemProfile)
print(mfScore)
# Get global bias
globalBias = modelAls.global_bias_
print(globalBias)
# Get bias of user
userBias = modelAls.user_bias_[0]
print(userBias)
# Get bias of item
itemBias = modelAls.item_bias_[indexFirstRec]
print(itemBias)
# Get total score
total = globalBias + userBias + itemBias + mfScore
print(total)
# As you can see, this is the same as the calculated score
print(firstRecScore)
