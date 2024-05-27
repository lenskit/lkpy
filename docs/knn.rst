k-NN Collaborative Filtering
============================

.. module:: lenskit.algorithms.knn

LKPY provides user- and item-based classical k-NN collaborative Filtering
implementations.  These lightly-configurable implementations are intended
to capture the behavior of the Java-based LensKit implementations to provide
a good upgrade path and enable basic experiments out of the box.

There are two different primary modes that you can use these algorithms in.  When using **explicit
feedback** (rating values), you usually want to use the defaults of weighted-average aggregation and
mean-centering normalization.  This is the default mode, and can be selected explicitly by passing
``feedback='explicit'`` to the class constructor.

With **implicit feedback** (unary data such as clicks and purchases, typically represented with
rating values of 1 for positive items), the usual design is sum aggregation and no centering. This
can be selected with ``feedback='implicit'``, which also configures the algorithm to ignore rating
values (when present) and treat every rating as 1::

    implicit_knn = ItemItem(20, feedback='implicit')

Attempting to center data on the same scale (all 1, for example) will typically produce invalid
results.  ItemKNN has diagnostics to warn you about this.

The ``feedback`` option only sets defaults; the algorithm can be further configured (e.g. to re-enable
rating values) with additional parameters to the constructor.

.. versionadded:: 0.14
    The ``feedback`` option and the ability to ignore rating values was added in LensKit 0.14.
    In previous versions, you need to specifically configure each option.

.. toctree::


Item-based k-NN
---------------

This is LensKit's item-based k-NN model, based on the description by
:cite:t:`deshpande:iknn`.

.. autoclass:: ItemItem
    :members:
    :show-inheritance:

User-based k-NN
---------------

.. autoclass:: UserUser
   :members:
   :show-inheritance:
