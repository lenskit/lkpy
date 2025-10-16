Configuration and Settings
==========================

.. py:module:: lenskit.config

LensKit supports reading configuration files (``lenskit.toml``,
``lenskit.local.toml``) for configuration.  This is automatically done by the
CLI, but can be manually done in your own programs as well.

Initializing Configuration
--------------------------

.. autofunction:: configure
    :alias: lenskit.configure

Accessing Configuration
-----------------------

.. autofunction:: lenskit_config

Configuration Model
-------------------

.. autoclass:: LenskitSettings()
    :exclude-members: model_config, settings_customize_sources

.. autoclass:: RandomSettings
    :exclude-members: model_config

.. autoclass:: PrometheusSettings
    :exclude-members: model_config

.. autoclass:: MachineSettings
    :exclude-members: model_config

.. autoclass:: TuneSettings
    :exclude-members: model_config
