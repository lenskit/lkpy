k-NN Collaborative Filtering
============================

LKPY provides user- and item-based classical k-NN collaborative Filtering
implementations.  These lightly-configurable implementations are intended
to capture the behavior of the Java-based LensKit implementations to provide
a good upgrade path and enable basic experiments out of the box.

There are two different primary that you can use these algorithms in.  When using **explicit
feedback** (rating values), you usually want to use the defaults of weighted-average aggregation and
mean-centering normalization.

With **implicit feedback** (unary data such as clicks and purchases, typically represented with
rating values of 1 for positive items), the usual design is sum aggregation and no centering::

    implicit_knn = ItemItem(20, center=False, aggregate='sum')

Attempting to center data on the same scale (all 1, for example) will typically produce invalid
results.  ItemKNN has diagnostics to warn you about this.

.. toctree::


Item-based k-NN
---------------

.. module:: lenskit.algorithms.item_knn

This is LensKit's item-based k-NN model, based on the description by :cite:t:`Deshpande2004-ht`.

.. autoclass:: ItemItem
    :members:
    :show-inheritance:

User-based k-NN
---------------

.. module:: lenskit.algorithms.user_knn

.. autoclass:: UserUser
   :members:
   :show-inheritance:
