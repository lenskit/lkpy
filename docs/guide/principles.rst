Design Goals and Principles
===========================

LensKit is intended to be a flexible toolkit for a wide range of teaching and
research purposes.  The LensKit paper :cite:p:`lkpy` documents several of these
principles in detail; as we have gained more experience, particularly with the
changes leading to LensKit 2025.1 (see :ref:`migrating`).

These principles capture the broad goals that guide LensKit development; they
are sometimes in tension with each other or other engineering concerns, and are
aspirational rather than fully-realized, but serious divergences are usually
bugs or work-in-progress.

Make best practices the default (or at least easy)
    It should be easy to do research using currently-understood best practices.

Use the pieces you want
    LensKit is a toolkit of individual pieces that do not have hard dependencies
    on each other.  If you have your own recommendation pipelines and want to
    use LensKit's metrics to evaluate recommendations you have saved in Parquet
    or CSV files, it works (and we encourage this use — standardizing metric
    implementations is important for research comparability, and using LensKit's
    metrics don't lock you in to the other parts of LensKit).  If you want to
    use the pipeline abstraction for components that work on complete different
    data sructures, that works too (we do this in POPROX_).  If you want to use
    some of the LensKit components but not others, that works too.

    The goal of LensKit is to support research using whatever computational
    toolkits you want.

Implement the pieces you need
    LensKit components are designed to be sufficiently configurable and
    re-composable to enable you to experiment with new recommendation ideas by
    only implementing the new pieces that are essential to your idea.

Bridge to other tools
    It should be easy to implement bridges to use scoring models from other
    libraries, or new scoring models using arbitrary computational libraries, in
    LensKit experiments (and, where appropriate, LensKit pipelines).

Use standard data structures
    This principle has evolved from its original version :cite:p:`lkpy`, when we
    used Pandas data frames as the interchange format between components, but
    LensKit data is still based on standard arrays (supporting both NumPy and
    PyTorch), sparse matrices, etc.

Support flexible data and paradigms
    LensKit intends to support a wide range of data and recommendation
    paradigms, including collaborative filters, content-based recommendation,
    session-based and session-aware recommendation, etc.

    .. note::

        This principle is more aspirational than the others at present.  For historical reasons,
        most of the models are collaborative filters, and they have the best support, but the
        :ref:`2025.1` changes and planned imminent work are partially designed to improve the
        for recommendation beyond user-item-matrix collaborative filtering.

.. _POPROX: https://docs.poprox.ai/reference/recommender/pipeline.html
