.. _component-impl:

Implementing Components
=======================

LensKit is particularly designed to excel in research and educational
applications, and for that you will often need to write your own components
implementing new scoring models, rankers, or other components. The
:ref:`pipeline design <pipeline>` and :ref:`standard pipelines
<standard-pipelines>` are intended to make this as easy as possible and allow
you to focus just on your logic without needing to implement a lot of
boilerplate like looking up user histories or ranking by score: you can
implement your training and scoring logic, and let LensKit do the rest.

Basics
~~~~~~

Implementing a component therefore consists of a few steps:

1.  Defining the configuration class.
2.  Defining the component class, with its ``config`` attribute declaration.
3.  Defining a ``__call__`` method for the component class that performs the
    component's actual computation.
4.  If the component supports training, implementing the
    :class:`~lenskit.training.Trainable` protocol by defining a
    :meth:`~lenskit.training.Trainable.train` method, or implement
    :ref:`iterative-training`.

A simple example component that computes a linear weighted blend of the scores
from two other components could look like this:

.. literalinclude:: examples/blendcomp.py

This component can be instantiated with its defaults:

.. testsetup::

    from blendcomp import LinearBlendScorer, LinearBlendConfig


.. doctest::

    >>> LinearBlendScorer()
    <LinearBlendScorer {
        "mix_weight": 0.5
    }>

You an instantiate it with its configuration class:

.. doctest::

    >>> LinearBlendScorer(LinearBlendConfig(mix_weight=0.2))
    <LinearBlendScorer {
        "mix_weight": 0.2
    }>

Finally, you can directly pass configuration parameters to the component constructor:

.. doctest::

    >>> LinearBlendScorer(mix_weight=0.7)
    <LinearBlendScorer {
        "mix_weight": 0.7
    }>


Component Configuration
~~~~~~~~~~~~~~~~~~~~~~~

As noted in the :ref:`pipeline documentation <component-config>`, components are
configured with *configuration objects*.  These are JSON-serializable objects
defined as Python dataclasses or Pydantic models, and define the different
settings or hyperparameters that control the model's behavior.

The choice of parameters are up to the component author, and each component will
have different configuration needs.  Some needs are common across many
components, though; see :ref:`config-conventions` for common LensKit
configuration conventions.

Component Operation
~~~~~~~~~~~~~~~~~~~

The heart of the component interface is the ``__call__`` method (components are
just callable objects).  This method takes the component inputs as parameters,
and returns the component's result.

Most components return an :class:`~lenskit.data.ItemList`.  Scoring components usually
have the following signature:

.. code:: python

    def __call__(self, query: QueryInput, items: ItemList) -> ItemList:
        ...

The ``query`` input receives the user ID, history, context, or other query
input; the ``items`` input receives the list of items to be scored (e.g., the
candidate items for recommendation).  The scorer then returns a list of scored
items.

Most component begin by converting the query to a
:class:`~lenskit.data.RecQuery`::

    def __call__(self, query: QueryInput, items: ItemList) -> ItemList:
        query = RecQuery.create(query)
        ...

It is conventional for scorers to return a copy of the input item list with the scores
attached, filling in ``NaN`` for items that cannot be scored.  After assembling a NumPy
array of scores, you can do this with::

    return ItemList(items, scores=scores)

Scalars can also be supplied, so if the scorer cannot score any of the items, it
can simply return a list with no scores::

    return ItemList(items, scores=np.nan)

Components do need to be able to handle items in ``items`` that were not seen
at training time.  If the component has saved the training item vocabulary, the
easiest way to do this is to use :meth:`~lenskit.data.ItemList.numbers`: with
``missing="negative"``::

    i_nums = items.numbers(vocabulary=self.items, missing="negative")
    scorable_mask = i_nums >= 0

Component Training
~~~~~~~~~~~~~~~~~~

Components that need to train models on training data must implement the
:class:`~lenskit.training.Trainable` protocol, either directly or through a
helper implementation like :class:`~lenskit.training.IterativeTraining`.  The
core of the ``Trainable`` protocol is the
:meth:`~lenskit.training.Trainable.train` method, which takes a
:class:`~lenskit.data.Dataset` and :class:`~lenskit.training.TrainingOptions`
and trains the model.

The details of training will vary significantly from model to model.  Typically,
though, it follows the following steps:

1.  Extract, prepare, and preprocess training data as needed for the model.
2.  Compute the model's parameters, either directly (i.e.
    :class:`~lenskit.basic.BiasScorer`) or through an optimization method (i.e.
    :class:`~lenskit.basic.ImplicitMFScorer`).
3.  Finalize the model parameters and clean up any temporary data.

Learned model parameters are then stored as attributes on the component class,
either directly or in a container object (such as a PyTorch
:class:`~torch.nn.Module`).

.. note::

    If the model is already trained and the
    :attr:`~lenskit.training.TrainingOptions.retrain` is ``False``, then the
    ``train`` method should return without any training.
    :class:`~lenskit.training.IterativeTraining` handles this automatically.

Further Reading
~~~~~~~~~~~~~~~

See :ref:`conventions` for more conventions for component design and configuration.
