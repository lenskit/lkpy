Algorithm Summary
=================

.. py:module:: lenskit.algorithms

LKPY provides general algorithmic concepts, along with implementations of several
algorithms.  These algorithm interfaces are based on the SciKit design patterns
:cite:p:`Buitinck2013-ks`, adapted for Pandas-based data structures.


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

Add-On Packages
~~~~~~~~~~~~~~~

See `add-on algorithms <addons.rst>`_ for additional algorithm families and bridges to other
packages.
