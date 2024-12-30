Scoring Models
==============

Most recommendation pipelines are built around a *scoring model* that scores
items for a recommendation query (e.g., user).  Standard top-*N* recommendation
uses these scores to rank items, and they can be used as inputs into other
techniques such as samplers and rerankers.  Scorers are almost always
:class:`~lenskit.pipeline.Trainable`, and by convention are named `XYZScorer`.

Scoring models are not limited to traditional pointwise scoring models such as
matrix factorization.  Many learning-to-rank models are also implemented as
scorers, but using a model optimized with a rank-based loss function.

Baseline Scorers
~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:

    lenskit.basic.BiasScorer
    lenskit.basic.PopScorer

Classical Collaborative Filters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:

    lenskit.knn.ItemKNNScorer
    lenskit.knn.UserKNNScorer
    lenskit.als.BiasedMFScorer
    lenskit.als.ImplicitMFScorer
    lenskit.svd.BiasedSVDScorer
