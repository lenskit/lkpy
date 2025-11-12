Other Components
================

LensKit provides a number of other components; they are detailed in the API reference,
but a few that are useful in assembling useful pipelines include:

Candidate Selectors
~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:

    lenskit.basic.TrainingItemsCandidateSelector
    lenskit.basic.UnratedTrainingItemsCandidateSelector
    lenskit.basic.AllTrainingItemsCandidateSelector

History Lookup
~~~~~~~~~~~~~~

LensKit pipelines use a history lookup component to obtain user profile data
when the recommender is called with only a user ID, so they do not need to
repeat that logic in each component.

.. autosummary::
    :nosignatures:

    lenskit.basic.UserTrainingHistoryLookup
