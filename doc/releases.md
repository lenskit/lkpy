# Release Notes

## 0.6.0 (In Progress)

See the [GitHub milestone](https://github.com/lenskit/lkpy/milestone/1) for a summary of what's happening!

- The `save` and `load` methods on algorithms have been removed.  Just pickle fitted models to save
  their data.  This is what SciKit does, see no need to deviate.
- Top-*N* evaluation has been completely revamped to make it easier to correctly implement and run
  evaluation metrics.  Batch recommend no longer attaches ratings to recommendations.
- Batch recommend & predict functions now take `nprocs` as keyword-only.
- Several bug fixes and testing improvements

### Internal Changes

These changes should not affect you if you are only consuming LensKit's algorithm and evaluation capabilities.

-   Rewrite the `CSR` class to be more ergonomic from Python, at the expense of making the NumPy jitclass
    indirect. It is available in the `.N` attribute.  Big improvement: it is now picklable.

## 0.5.0

LensKit 0.5.0 modifies the algorithm APIs to follow the SciKit design patterns instead of
our previous custom patterns.  Highlights of this change:

- Algorithms are trained in-place — we no longer have distinct model objects.
- Model data is stored as attributes on the algorithm object that end in `_`.
- Instead of writing `model = algo.train_model(ratings)`, call `algo.fit(ratings)`.

We also have some new capabilities:

- Ben Frederickson's Implicit library

## 0.3.0

A number of improvements, including replacing Cython/OpenMP with Numba and adding ALS.

## 0.2.0

A lot of fixes to get ready for RecSys.

## 0.1.0

Hello, world!
