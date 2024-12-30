.. _rankers:

Ranking Algorithms
==================

LensKit provides several ranking components that rank items.

The usual design for a ranker is to take as input a list of items named `items`,
usually with scores (typically the output of a :ref:`scoring model <scorers>`),
and possibly the query, and return an ordered item list.  Rankers can also take
ranked inputs for re-ranking.  By convention, they are named ``XYZRanker``.

Top-N Ranking
~~~~~~~~~~~~~

Classic top-*N* ranking is provided by :class:`lenskit.basic.TopNRanker`. The
standard pipelines configured by :class:`~lenskit.pipeline.RecPipelineBuilder`
use this ranker by default.

Stochastic Ranking
~~~~~~~~~~~~~~~~~~

- :class:`lenskit.basic.SoftmaxRanker` computes a randomized Plackett-Luce
  ranking, where each item is selected with probability proportional to its
  score.
- :class:`lenskit.basic.RandomSelector` selects items uniformly at random.
