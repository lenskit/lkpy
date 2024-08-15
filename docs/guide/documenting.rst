Documenting Experiments
=======================

.. todo::
    This chapter needs to be rewritten for :ref:`2024.1`.

When publishing results — either formally, through a venue such as ACM Recsys,
or informally in your organization, it's important to clearly and completely
specify how the evaluation and algorithms were run.

Since LensKit is a toolkit of individual pieces that you combine into your
experiment however best fits your work, there is not a clear mechanism for
automating this reporting.  This document is to provide guidance for what and
how you should report, and how it maps to the LensKit code you are using.

Common Evaluation Problems Checklist
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This checklist is to help you make sure that your evaluation and results are
accurately reported.

* Pass `include_missing=True` to :py:meth:`~lenskit.topn.RecListAnalysis.compute`. This
  operation defaults to `False` for compatiability reasons, but the default will
  change in the future.

* Correctly fill missing values from the evaluation metric results.  They are
  reported as `NaN` (Pandas NA) so you can distinguish between empty lists and
  lists with no relevant items, but should be appropraitely filled before
  computing aggregates.

* Pass `k` to :py:meth:`~lenskit.topn.RecListAnalysis.add_metric` with the
  target list length for your experiment.  LensKit cannot reliably detect how
  long you intended to make the recommendation lists, so you need to specify the
  intended length to the metrics in order to correctly account for it.

Reporting Algorithms
~~~~~~~~~~~~~~~~~~~~

.. note::

    This section is still in progress.

You need to clearly report the algorithms that you have used along with their
hyperparameters.

The algorithm name should be the name of the class that you used (e.g.
:py:class:`~lenskit.algorithms.knn.ItemItem`). The hyperparameters are the
options specified to the constructor, except for options that only affect
algorithn peformance but not behavior.

For example:

+------------+-------------------------------------------------------------------------------+
| Algorithm  |                                Hyperparameters                                |
+============+===============================================================================+
| ItemItem   | :math:`k_\mathrm{max}=20, k_\mathrm{min}=2, s_\mathrm{min}=1.0\times 10^{-3}` |
+------------+-------------------------------------------------------------------------------+
| ImplicitMF | :math:`k=50, \lambda_u=0.1, \lambda_i=0.1, w=40`                              |
+------------+-------------------------------------------------------------------------------+

If you use a top-N implementation other than the default
:py:class:`~lenskit.algorithms.basic.TopN`, or reconfigure its candidate
selector, also clearly document that.

Reporting Experimental Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The important things to document are the data splitting strategy, the target
recommendation list length, and the candidate selection strategy (if something
other than the default of all unrated items is used).

If you use one of LensKit's built-in splitting strategies from :py:class:`~lenskit.crossfold`
without modification, report:

- The splitting function used.
- The number of partitions or test samples.
- The number of users per sample (when using
  :py:class:`~lenskit.crossfold.sample_users`) or ratings per sample (when using
  :py:class:`~lenskit.crossfold.sample_ratings`).
- When using a user-based strategy (either
  :py:class:`~lenskit.crossfold.partition_users` or
  :py:class:`~lenskit.crossfold.sample_users`), the test rating selection
  strategy (class and parameters), e.g. ``SampleN(5)``.

Any additional pre-processing (e.g. filtering ratings) should also be clearly
described.  If additional test pre-processing is done, such as removing ratings
below a threshold, document that as well.

Since the experimental protocol is implemented directly in Python code,
automated reporting is not practical.

Reporting Metrics
~~~~~~~~~~~~~~~~~

Reporting the metrics themelves is relatively straightforward.  The
:py:meth:`lenskit.topn.RecListAnalysis.compute` method will return a data frame
with a metric score for each list.  Group those by algorithm and report the
resulting scores (typically with a mean).

The following code will produce a table of algorithm scores for hit rate, nDCG
and MRR, assuming that your algorithm identifier is in a column named ``algo``
and the target list length is in ``N``::

    rla = RecListAnalysis()
    rla.add_metric(topn.hit, k=N)
    rla.add_metric(topn.ndcg, k=N)
    rla.add_metric(topn.recip_rank, k=N)
    scores = rla.compute(recs, test, include_missing=True)
    # empty lists will have na scores
    scores.fillna(0, inplace=True)
    # group by agorithm
    algo_scores = scores.groupby('algorithm')[['hit', 'ndcg', 'recip_rank']].mean()
    algo_scores = algo_scores.rename(columns={
        'hit': 'HR',
        'ndcg': 'nDCG',
        'recip_rank': 'MRR'
    })

You can then use :py:meth:`pandas.DataFrame.to_latex` to convert ``algo_scores``
to a LaTeX table to include in your paper.

Citing LensKit
~~~~~~~~~~~~~~

Finally, cite [LKPY]_ as the package used for producing and/or evaluating
recommendations.

.. [LKPY]
    Michael D. Ekstrand. 2020.
    LensKit for Python: Next-Generation Software for Recommender Systems Experiments.
    In <cite>Proceedings of the 29th ACM International Conference on Information and Knowledge Management</cite> (CIKM '20).
    DOI:`10.1145/3340531.3412778 <https://dx.doi.org/10.1145/3340531.3412778>`_.
    arXiv:`1809.03125 <https://arxiv.org/abs/1809.03125>`_ [cs.IR].

.. code-block:: bibtex

    @INPROCEEDINGS{lkpy,
    title           = "{LensKit} for {Python}: Next-Generation Software for
                        Recommender System Experiments",
    booktitle       = "Proceedings of the 29th {ACM} International Conference on
                        Information and Knowledge Management",
    author          = "Ekstrand, Michael D.",
    year            =  2020,
    url             = "http://dx.doi.org/10.1145/3340531.3412778",
    conference      = "CIKM '20",
    doi             = "10.1145/3340531.3412778"
    pages           = "2999--3006"
    }
