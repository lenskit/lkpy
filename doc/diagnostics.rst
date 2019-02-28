Errors and Diagnostics
======================

Logging
~~~~~~~

LensKit algorithms and evaluation routines report diagnostic data using the standard Python
:py:mod:`logging` framework.  Loggers are named after the corresponding Python module, and all
live under the ``lenskit`` namespace.

**Algorithms** usually report erroneous or anomalous conditions using Python exceptions and
warnings.  **Evaluation code**, such as that in :py:mod:`lenskit.batch`, typically reports
such conditions using the logger, as the common use case is to be running them in a script.

Warnings
~~~~~~~~

In addition to Python standard warning types such as :py:class:`warnings.DeprecationWarning`,
LensKit uses the following warning classes to report anomalous problems in
use of LensKit.

.. autoclass:: lenskit.DataWarning
