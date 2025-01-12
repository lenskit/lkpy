Model Implementation Tips
=========================

Implementing algorithms is fun, but there are a few things that are good to keep in mind.

In general, development follows the following:

1. Correct
2. Clear
3. Fast

In that order.  Further, we always want LensKit to be *usable* in an easy
fashion.  Code implementing commonly-used models, however, may be quite complex
in order to achieve good performance.

.. _iterative-training:

Iterative Training
~~~~~~~~~~~~~~~~~~

The :class:`lenskit.training.IterativeTraining` class provides a standardized
interface and training loop support for training models with iterative methods
that pass through the training data in multiple *epochs*.  Models that use this
support extend :class:`~lenskit.training.IterativeTraining` in addition to
:class:`~lenskit.pipeline.Component`, and implement the
:meth:`~lenskit.training.IterativeTraining.training_loop` method instead of
:meth:`~lenskit.training.Trainable.train`.  Iteratively-trainable components
should also have an ``epochs`` setting on their configuration class that
specifies the number of training epochs to run.

The :meth:`~lenskit.training.IterativeTraining.training_loop` method does 3 things:

1.  Set up initial data structures, preparation, etc. needed for model training.
2.  Train the model, yielding after each training epoch.  It can optionally
    yield a set of metrics, such as training loss or update magnitudes.
3.  Perform any final steps and training data cleanup.

The model should be usable after each epoch, to support things like measuring
performance on validation data.

The training loop itself is represented as a Python iterator, so that a ``for``
loop will loop through the training epochs.  While the interface definition
specifies the ``Iterator`` type in order to minimize restrictions on component
implementers, we recommend that it actually be a ``Generator``, which allows the
caller to request early termination (through the
:meth:`~collections.abc.Generator.close` method).  We also recommend that the
``training_loop()`` method only return the generator after initial data preparation
is complete, so that setup time is not included in the time taken for the first
loop iteration.  The easiest way to do implement this is by delegating to an
inner loop function, written as a Python generator:

.. code:: python

    def training_loop(self, data: Dataset, options: TrainingOptions):
        # do initial data setup/prep for training
        context = ...
        # pass off to inner generator
        return self._training_loop_impl(context)

    def _training_loop_impl(self, context):
        for i in range(self.config.epochs):
            # do the model training
            # compute the metrics
            try:
                yield {'loss': loss}
            except GeneratorExit:
                # client code has requested early termination
                break

        # any final cleanup steps
