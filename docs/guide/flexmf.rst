.. _flexmf:

Flexible Matrix Factorization
=============================

.. currentmodule:: lenskit.flexmf


.. stability:: experimental

    The FlexMF model framework is currently provided as an experimental preview.
    It works pretty well, but may be adjusted as we stabilize it and gain more
    experience in the next months.


The LensKit ``FlexMF`` (*Flexible Matrix Factorization*) family of models use
matrix factorization in various configurations to realize several scoring models
from the literature in a single configurable design, implemented in PyTorch with
support for GPU-based training.

The FlexMF components and configuration are in the :mod:`lenskit.flexmf` package.

First Model
~~~~~~~~~~~

FlexMF works like any other LensKit scorer.  To train a simple implicit-feedback
scorer with logistic matrix factorization :cite:p:`LogisticMF`, you can do:

>>> from lenskit.flexmf import FlexMFImplicitScorer
>>> from lenskit.data import load_movielens
>>> from lenskit import topn_pipeline, recommend
>>> # load movie data
>>> data = load_movielens('data/ml-latest-small')
>>> # set up model
>>> model = FlexMFImplicitScorer(embedding_size=32, loss="logistic")
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

.. _flexmf-common-config:

Common Configuration
~~~~~~~~~~~~~~~~~~~~

All FlexMF models share some configuration option in common, defined by
:class:`FlexMFConfigBase`:

Model Structure Options
    ``embedding_size``
        The dimension of the matrix factorization.  FlexMF works best when
        this is a power of 2.

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

    .. tip::

        Our internal experiments have generally found ``AdamW`` regularization to be
        more effective, and seen little to no benefit from the sparse gradients
        allowed by ``L2``.  We may remove configurable regularization types before
        removing the experimental label from FlexMF.

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
prediction model with biased matrix factorization.  The model itself is
mathematically identical to :class:`~lenskit.als.BiasedMFScorer`, but is trained
using minibatch gradient descent in PyTorch and can use a GPU.  User and item
biases are learned jointly with the embeddings, and are attenuated for
low-information users and items through regularization instead of an explicit
damping term.

Implicit Feedback
~~~~~~~~~~~~~~~~~

:class:`FlexMFImplicitScorer` provides the implicit-feedback scorers in FlexMF.
This scorer supports multiple loss functions and training regimes that can be
selected to yield logistic matrix factorization, BPR, WARP, or others (see
:class:`FlexMFImplicitConfig` for full options).

Presets
-------

For easy configuration, FlexMF provides *presets* (the
:attr:`~FlexMFImplicitConfig.preset` option) that set the defaults for other
options to match the originally-published versions of the models FlexMF can
realize.  Select a preset by configuring e.g. ``preset="bpr"``.

Below are the settings controlled by each preset:

+--------------------+----------+-----------+----------+
| Preset             | bpr      | warp      | lightgcn |
+====================+==========+===========+==========+
| loss               | pairwise | warp      | pairwise |
+--------------------+----------+-----------+----------+
| negative_strategy  | uniform  | misranked | uniform  |
+--------------------+----------+-----------+----------+
| user_bias          | False    | False     | False    |
+--------------------+----------+-----------+----------+
| item_bias          | False    | False     | False    |
+--------------------+----------+-----------+----------+
| convolution_layers | 0        | 0         | 3        |
+--------------------+----------+-----------+----------+

These are only defaults — settings in :ref:`flexmf-common-config` and
:ref:`flexmf-implicit-config` will override these settings.

.. _flexmf-implicit-config:

Additional Configuration Options
--------------------------------

The two primary options that control the model's overall behavior are the *loss
function* and the *sampling strategy*.

Three loss functions (:attr:`~FlexMFImplicitConfig.loss`) are supported:

``"logistic"``
    Logistic loss, as used in Logistic Matrix Factorization :cite:p:`LogisticMF`.
``"pairwise"``
    Pairwise rank loss, as used in Bayesian Personalized Ranking :cite:p:`BPR`.
``"warp"``
    Weighted approximate rank lost, a revised version of pairwise loss used in
    WARP :cite:p:`warp-korder`.  Only works with the ``"misranked"`` sampling
    strategy.

Three sampling strategies (:attr:`~FlexMFImplicitConfig.negative_strategy`) are
supported:

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

Finally, neighborhood aggregation in the style of LightGCN :cite:p:`LightGCN` can
be enabled with the :attr:`~FlexMFImplicitConfig.convolution_layers` option.

Relationship to Published Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FlexMF is designed to provide reasonable implementations of various designs from
the literature in an integrated manner, differing primarily in fundamental
design points rather than implementation details, and facilitating good code
re-use and testing across implementations.

-   The explicit-feedback model implements **biased matrix factorization** as
    described in many papers, with the core model appearing in both ALS
    :cite:p:`ExplicitALS` and FunkSVD :cite:p:`FunkSVD`. The description most
    closely aligning with this implementation is probably that of
    :cite:t:`KorenMF`.
-   The default implicit settings are similar to **logistic matrix
    factorization** :cite:p:`LogisticMF`.  It is not an exact implementation,
    because implicit-feedback FlexMF uses negative sampling whereas
    :cite:t:`LogisticMF` trained on entire users with different weights for
    positive and negative items.
-   **Bayesian Personalized Ranking** with matrix factorization (BPR-MF)
    :cite:p:`BPR` uses the pairwise loss and uniform negative sampling; these
    are the default with the ``"bpr"`` preset.
-   **Weighted Approximate Rank Loss** (WARP) :cite:p:`WARP,warp-korder` uses
    *misranked* negative sampling and WARP loss.  These are the default with the
    ``"warp"`` preset.
-   **Light Graph Convolutional Networks** (LightGCN) :cite:p:`LightGCN`
    modifies the model, not the loss, by adding multiple layers of aggregated
    neighbor embeddings; it can be used with any supported loss or negative
    strategy.  LightGCN is enabled by the
    :attr:`~FlexMFImplicitConfig.convolution_layers` setting, where a positive
    number of layers puts the model into LightGCN mode.  The ``"lightgcn"``
    preset defaults to 2 layers with pairwise loss.  The mixing coefficients are
    not currently configurable, as :cite:t:`LightGCN` found that just averaging
    the embedding layers worked pretty well.
