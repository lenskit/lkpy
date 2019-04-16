Algorithm Implementation Tips
=============================

Implementing algorithms is fun, but there are a few things that are good to keep in mind.

Pickling and Sharing
--------------------

LensKit uses Python pickling (or JobLib's modified pickling in :py:func:`joblib.dump`) quite
a bit to save and reload models and to share model data between concurrent processes.  This
generally just works, and you don't need to implement any particular save/load logic in order
to have your algorithm be savable and sharable.

There are a few exceptions, though.

**If your algorithm updates state after fitting**, this should *not* be pickled.  An example of
this would be caching predictions or recommendations to save time in subsequent calls.  Only the
model parameters and estimated parameters should be pickled.  If you have caches or other
ephemeral structures, override ``__getstate__`` and ``__setstate__`` to exclude them from the
saved data and to initialize caches to empty values on unpickling.

Memory Map Friendliness
-----------------------

LensKit uses :py:cls:`joblib.Parallel` to parallelize internal operations (when it isn't using Numba).
Joblib is pretty good about using shared memory to minimize memory overhead in parallel computations,
and LensKit has some tricks to maximize this use. However, it does require a bit of attention in
your algorithm implementation.

The easiest way to make this fail is to use many small NumPy or Pandas data structures.  If you have
a dictionary of :py:cls:`np.ndarray` objects, for instance, it will cause a problem.  This is because
each array will be memory-mapped, and each map will *reopen* the file.  Having too many active
open files will cause your process to run out of file descriptors on many systems.  Keep your
object count to a small, ideally fixed number; in :py:cls:`lenskit.algorithms.basic.UnratedItemSelector`,
we do this by storing user and item indexes along with a :py:cls:`matrix.CSR` containing the items
rated by each user.  The old implementation had a dictionary mapping user IDs to ``ndarray``s with
each user's rated items.  This is a change from :math:`|U|+1` arrays to 5 arrays.
