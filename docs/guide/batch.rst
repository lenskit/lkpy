.. _batch::

Batch-Running Recommenders
==========================

.. highlight:: python

The functions in :mod:`lenskit.batch` enable you to generate many recommendations or
predictions at the same time, useful for evaluations and experiments.

The batch functions can parallelize over users with the optional ``n_jobs`` parameter, or
the ``LK_NUM_PROCS`` environment variable.

.. note::
    Scripts calling the batch recommendation or prediction facilites must be *protected*;
    that is, they should not directly perform their work when run, but should define functions
    and call a ``main`` function when run as a script, with a block like this at the end of the
    file::

        def main():
            # do the actual work

        if __name__ == '__main__':
            main()

    If you are using the batch functions from a Jupyter notebook, you should be fine - the
    Jupyter programs are appropriately protected.
