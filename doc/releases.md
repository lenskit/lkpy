# Release Notes

## 0.6.0 (In Progress)

See the [GitHub milestone](https://github.com/lenskit/lkpy/milestone/1) for a summary of what's happening!

- Several bug fixes and testing improvements

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
