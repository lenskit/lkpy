Algorithm Summary
=================

.. py:module:: lenskit.algorithms

LKPY provides general algorithmic concepts, along with implementations of several
algorithms.  These algorithm interfaces are based on the SciKit design patterns
:cite:p:`sklearn-api`, adapted for Pandas-based data structures.


All algorithms implement the `standard interfaces`_.

.. _standard interfaces: interfaces.html

Basic Algorithms
~~~~~~~~~~~~~~~~

.. autosummary::

    bias.Bias
    basic.PopScore
    basic.TopN
    basic.Fallback
    basic.UnratedItemCandidateSelector
    basic.Memorized

k-NN Algorithms
~~~~~~~~~~~~~~~

.. autosummary::

    knn.UserUser
    knn.ItemItem

Matrix Factorization
~~~~~~~~~~~~~~~~~~~~

.. autosummary::

    als.BiasedMF
    als.ImplicitMF

Add-On Packages
~~~~~~~~~~~~~~~

See `add-on algorithms <addons.rst>`_ for additional algorithm families and bridges to other
packages.
