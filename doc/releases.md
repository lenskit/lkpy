# Release Notes

## 0.8.0 (in progress)

See the [GitHub milestone](https://github.com/lenskit/lkpy/milestone/5) for full change list.

### Algorithm Updates

- Added a coordinate descent solver to explicit-feedback ALS and made it the default.  The old
  LU-based solver is still available with `method='lu'`.
- Added a conjugate gradient solver to implicit-feedback ALS and made it the default.

## 0.7.0

See the [GitHub milestone](https://github.com/lenskit/lkpy/milestone/3) for full change list.

- Use [Joblib][] for parallelism in batch routines.
- `nprocs` arguments are renamed to `n_jobs` for consistency with Joblib.
- Removed `parallel` option on `MultiEval` algorithms, as it was unused.
- Made `MultiEval` default to using each recommender's default candidate
  set, and adapt algorithms to recommenders prior to evaluation.
- Make `MultiEval` require named arguments for most things.
- Add support to `MultiEval` to save the fit models.
- `RecListAnalysis` can optionally ensure all test users are returned, even
  if they lack recommendation lists.
- Performance improvements to algorithms and evaluation.

[Joblib]: https://joblib.readthedocs.io


## 0.6.1

See the [GitHub milestone](https://github.com/lenskit/lkpy/milestone/4) for full change list.

- Fix inconsistency in both code and docs for recommend list sizes for top-*N* evaluation.
- Fix user-user to correctly use `sum` aggregate.
- Improve performance and documentation

## 0.6.0

See the [GitHub milestone](https://github.com/lenskit/lkpy/milestone/1) for a summary of what's happening!

- The `save` and `load` methods on algorithms have been removed.  Just pickle fitted models to save
  their data.  This is what SciKit does, we see no need to deviate.
- The APIs and model structures for top-*N* recommendation is reworked to enable algorithms to
  produce recommendations more automatically.  The `Recommender` interfaces now take a `CandidateSelector`
  to determine default candidates, so client code does not need to compute candidates on their own.
  One effect of this is that the `batch.recommend` function no longer requires a candidate selector,
  and there can be problems if you call `Recommender.adapt` before fitting a model.
- Top-*N* evaluation has been completely revamped to make it easier to correctly implement and run
  evaluation metrics.  Batch recommend no longer attaches ratings to recommendations.  See 
  [Top-N evaluation](evaluation/topn-metrics.html) for details.
- Batch recommend & predict functions now take `nprocs` as a keyword-only argument.
- Several bug fixes and testing improvements.

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
