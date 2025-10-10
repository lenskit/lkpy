.. _flexmf:

Flexible Matrix Factorization
=============================

.. currentmodule:: lenskit.flexmf


.. stability:: experimental

    The FlexMF model framework is currently provided as an experimental preview.
    It works pretty well, but may be adjusted as we stabilize it and gain more
    experience in the next months.


Since :ref:`2025.3.0`, LensKit provides a family of embedding-based scoring
models called ``FlexMF`` (*Flexible Matrix Factorization*).  These models use
matrix factorization in various configurations to realize several scoring models
from the literature in a single configurable design, implemented in PyTorch with
support for GPU-based training.

The FlexMF components and configuration are in the :mod:`lenskit.flexmf` package.

First Model
~~~~~~~~~~~

FlexMF works like any other LensKit scorer.  To train a simple implicit-feedback
scorer with logistic matrix factorization, you can do:

>>> from lenskit.flexmf import FlexMFImplicitScorer
>>> from lenskit.data import load_movielens
>>> from lenskit import topn_pipeline, recommend
>>> # load movie data
>>> data = load_movielens('data/ml-latest-small')
>>> # set up model
>>> model = FlexMFImplicitScorer(embedding_size="50", loss="logistic")
>>> pipe = topn_pipeline(model, n=10)
>>> # train the model
>>> pipe.train(data)
>>> # recommend for user 500
>>> recommend(pipe, 500)
<ItemList of 10 items with 2 fields {
  ids: ...
  numbers: [...]
  rank: ...
  score: [...]
}>

Common Configuration
~~~~~~~~~~~~~~~~~~~~

All FlexMF models share some configuration option in common, defined by
:class:`FlexMFConfigBase`:

Model Structure Options
    ``embedding_size``
        The dimension of the matrix factorization.

Regularization Options
    FlexMF supports two different forms of regularization:
    :class:`~torch.optim.AdamW` weight decay and L2 regularization.  With L2
    regularization, the term is included directly in the loss function, and
    the model can be trained with sparse gradients.

    ``reg_method``:
        The method to use for regularization (``AdamW`` or ``L2``), or ``None``
        to disable regularization.

    ``regularization``:
        The regularization weight.

Training Options
    ``batch_size``:
        The size for individual training batches.  The optimal batch size is usually
        much larger than deep models, because the collaborative filtering models are
        relatively simple.
    ``learning_rate``:
        The base learning rate for the :class:`~torch.optim.AdamW` or
        :class:`~torch.optim.SparseAdam` optimizer.
    ``epochs``:
        The number of training epochs.


Explicit Feedback
~~~~~~~~~~~~~~~~~

The :class:`FlexMFExplicitScorer` class provides an explicit-feedback rating
prediction model with biased matrix factorization.  The model itself is the same
as that used by :class:`~lenskit.als.BiasedMFScorer`, but is trained using
minibatch gradient descent in PyTorch and can train a GPU.  User and item biases
are learned jointly with the embeddings, and are attenuated for low-information
users and items through regularization instead of an explicit damping term.

Implicit Feedback
~~~~~~~~~~~~~~~~~

:class:`FlexMFImplicitScorer` provides the implicit-feedback scorers in FlexMF.
This scorer supports multiple loss functions and training regimes that can be
selected to yield logistic matrix factorization, BPR, WARP, or others (see
:class:`FlexMFImplicitConfig` for full options).

The two primary options that control the model's overall behavior are the *loss
function* and the *sampling strategy*.

Three loss functions (``loss``) are supported:

``"logistic"``
    Logistic loss, as used in Logistic Matrix Factorization :cite:p:`LogisticMF`.
``"pairwise"``
    Pairwise rank loss, as used in Bayesian Personalized Ranking :cite:p:`BPR`.
``"warp"``
    Weighted approximate rank lost, a revised version of pairwise loss used in
    WARP :cite:p:`warp-korder`.  Only works with the ``"misranked"`` sampling
    strategy.

Three sampling strategies (``negative_strategy``) are supported:

``"uniform"``
    Negative items are sampled uniformly at random from the corpus.  This is the
    default for logistic and pairwise losses.
``"popular"``
    Negative items are sampled proportional to their popularity in the training
    data.
``"misranked"``
    Negative items are sampled based on their scores from the model so far, looking
    for misranked items.  This strategy comes from WARP, but can be used with other
    loss functions as well.  It is the default (and only) strategy for WARP loss.

You can combine these to realize several designs from the literature:

-   **Logistic matrix factorization** :cite:p:`LogisticMF` by using
    ``loss="logistic"`` with any sampling strategy.  This implementation differs
    slightly from the paper in that it uses negative sampling instead of
    training on an entire user.

-   Classic **Bayesian Personalized Ranking** :cite:p:`BPR` by using
    ``loss="pairwise"`` with the ``negative_strategy="uniform"``.  The negative
    strategy can also be changed, although we have rarely seen ``"popular"`` be
    effective.

-   **Weighted Approximate Rank Loss** (WARP) :cite:p:`WARP,warp-korder` by using
    ``loss="warp"``.
