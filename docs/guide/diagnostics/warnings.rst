.. _warnings:

LensKit Warnings
================

This page documents some of the warnings LensKit emits, with explanations (and
sometimes solutions).

PyTorch-related warnings
~~~~~~~~~~~~~~~~~~~~~~~~

.. _LKW-TCOMP:

TCOMP: ``torch.compile`` failures
--------------------------------

The ``TCOMP`` LensKit warning is issued when :func:`torch.compile` fails to
compile one of LensKit's PyTorch models.  In this case, LensKit will use the
model without Inductor compilation.  The model will still work, but performance
may be reduced.

There are several reasons ``torch.compile`` may fail:

- Running LensKit on Windows (the toolchain needed for working compilation on
  Windows is more difficult to set up, so LensKit proactively disabled
  compilation on Windows).
- Using PyTorch older than 2.8 (some LensKit models use features that could not
  be compiled in older PyTorch versions).
- Running LensKit on a version of Python that PyTorch hasn't added compilation
  support for yet (as of Oct. 2025, Python 3.14 is in this category).
