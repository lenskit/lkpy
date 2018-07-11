Algorithm Interfaces
====================

.. module:: lenskit.algorithms

LKPY's batch routines and utility support for managing algorithms expect algorithms
to implement consistent interfaces.  This page describes those interfaces.

The interfaces are realized as abstract base classes with the Python :py:mod:`abc` module.
Implementations must be registered with their interfaces, either by subclassing the interface
or by calling :py:meth:`abc.ABCMeta.register`.

Rating Prediction
-----------------

.. autoclass:: Predictor
   :members:


Model Training
--------------

Most algorithms have some concept of a trained model.  The ``Trainable`` interface captures the
ability of a model to be trained and saved to disk.

.. autoclass:: Trainable
   :members:
