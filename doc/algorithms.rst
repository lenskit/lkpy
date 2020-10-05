Algorithm Summary
=================

.. py:module:: lenskit.algorithms

LKPY provides general algorithmic concepts, along with implementations of several
algorithms.  These algorithm interfaces are based on the SciKit design patterns
[SKAPI]_, adapted for Pandas-based data structures.


All algorithms implement the `standard interfaces`_.

.. _standard interfaces: interfaces.html

Basic Algorithms
~~~~~~~~~~~~~~~~

.. autosummary::

    bias.Bias
    basic.Popular
    basic.TopN
    basic.Fallback
    basic.UnratedItemCandidateSelector
    basic.Memorized

k-NN Algorithms
~~~~~~~~~~~~~~~

.. autosummary::

    user_knn.UserUser
    item_knn.ItemItem

Matrix Factorization
~~~~~~~~~~~~~~~~~~~~

.. autosummary::

    als.BiasedMF
    als.ImplicitMF
    funksvd.FunkSVD

TensorFlow
~~~~~~~~~~

.. autosummary::

    tf.BiasedMF
    tf.IntegratedBiasMF
    tf.BPR

External Library Wrappers
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    
    implicit.BPR
    implicit.ALS
    hpf.HPF

References
~~~~~~~~~~

.. [SKAPI] Lars Buitinck, Gilles Louppe, Mathieu Blondel, Fabian Pedregosa, Andreas Mueller,
    Olivier Grisel, Vlad Niculae, Peter Prettenhofer, Alexandre Gramfort, Jaques Grobler,
    Robert Layton, Jake Vanderplas, Arnaud Joly, Brian Holt, and Gaël Varoquaux. 2013.
    API design for machine learning software: experiences from the scikit-learn project.
    arXiv:`1309.0238 <http://arxiv.org/abs/1309.0238>`_ [cs.LG].
