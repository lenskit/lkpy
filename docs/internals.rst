LensKit Internals
-----------------

These modules are primarily for internal infrastructural support in Lenskit.
Neither LensKit users nor algorithm developers are likely to need to use this
code directly.

.. class:: lenskit.types.RandomSeed

    Random seed values for LensKit models and components.  Can be any valid
    input to :func:`seedbank.numpy_rng`, including:

    * Any :data:`seedbank.SeedLike`
    * A :class:`numpy.random.Generator`
    * A :class:`numpy.random.RandomState` (deprecated)

    .. note::

        This is a type alias, not a class; it is documented as a class to work
        around limitations in Sphinx.

.. autoclass:: lenskit.types.UITuple
