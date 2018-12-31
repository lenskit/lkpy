k-NN Collaborative Filtering
============================

LKPY provides user- and item-based classical k-NN collaborative Filtering
implementations.  These lightly-configurable implementations are intended
to capture the behavior of the Java-based LensKit implementations to provide
a good upgrade path and enable basic experiments out of the box.

.. contents:: :toc:

Item-based k-NN
---------------

.. module:: lenskit.algorithms.item_knn

.. autoclass:: ItemItem
    :members:
    :show-inheritance:

.. autoclass:: IIModel

User-based k-NN
---------------

.. module:: lenskit.algorithms.user_knn

.. autoclass:: UserUser
   :members:
   :show-inheritance:
