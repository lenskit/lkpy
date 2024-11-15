.. _batch:

Batch-Running Pipelines
=======================

.. py:currentmodule:: lenskit.batch

.. highlight:: python

The functions and :py:class:`BatchPipelineRunner` class in
:py:mod:`lenskit.batch` enable you to generate many recommendations or
predictions at the same time, useful for evaluations and experiments.

.. admonition:: Import Protection
    :class: important

    Scripts using batch pipeline operations must be *protected*; that is, they
    should not directly perform their work when run, but should define functions
    and call a ``main`` function when run as a script, with a block like this at
    the end of the file::

        def main():
            # do the actual work

        if __name__ == '__main__':
            main()

    If you are using the batch functions from a Jupyter notebook, you should be fine - the
    Jupyter programs are appropriately protected.
