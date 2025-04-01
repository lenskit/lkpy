Model State
===========

.. py:module:: lenskit.state

The :mod:`lenskit.state` module provides support code for managing the state of
components built on machine learning models, such as extrating learned parameters
and saving model checkpoints.

These features are modeled after the PyTorch ``state_dict`` design: components
with state should implement :class:`ParameterContainer` and expose their state
as dictionaries.  This state can then be saved and loaded, including in
pickle-safe formats such as ``zarr`` or ``safetensors``.

.. autosummary::
    :toctree: .
    :nosignatures:

    ~lenskit.state.ParameterContainer
