Getting Started
===============

We're working on documentation!

For now, this example computes nDCG for an item-based k-NN collaborative filter::

    import pandas as pd
    from lenskit import batch, topn
    from lenskit import crossfold as xf
    from lenskit.algorithms import item_knn as knn

    ratings = pd.read_csv('ml-100k/u.data', sep='\t',
            names=['user', 'item', 'rating', 'timestamp'])

    algo = knn.ItemItem(30)

    def eval(train, test):
        model = algo.train(train)
        users = test.user.unique()
        recs = batch.recommend(algo, model, users, 100,
                topn.UnratedCandidates(train))
        # combine with test ratings for relevance data
        res = pd.merge(recs, test, how='left',
                    on=('user', 'item'))
        # fill in missing 0s
        res.loc[res.rating.isna(), 'rating'] = 0
        return res

    # compute evaluation
    splits = xf.partition_users(ratings, 5,
            xf.SampleFrac(0.2))
    recs = pd.concat((eval(train, test)
                    for (train, test) in splits))

    # compile results
    ndcg = recs.groupby('user').rating.apply(topn.ndcg)
