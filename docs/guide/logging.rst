Logging and Progress Reporting
==============================

.. _Rich: https://rich.readthedocs.io/

.. py:currentmodule:: lenskit.logging

LensKit provides support code for logging and progress reporting.  This code
lives in the :py:mod:`lenskit.logging` package and provides several
capabilities:

-   Backend-independent progress reporting, with colorful progress bars (via
    Rich_) on terminals.
-   Easy logging configuration for recommender scripts, supporting log files and
    verbosity options, along with environment variables for logging levels.
-   Integration between :py:mod:`structlog` and Python :py:mod:`logging`.
-   Tracking resource consumption for recommender experiment tasks (via :class:`Task`).
-   Collecting log output from subprocesses in process-based parallel computation.

Logging Configuration
~~~~~~~~~~~~~~~~~~~~~

When writing code that uses LensKit, such as experiment runners, the
:class:`LoggingConfig` class provides an easy way to set up logging for
common command-line scenarios.  Common usage looks like this (where ``options``
is a variable of command-line options from :mod:`argparse` or similar):

.. code:: python

    lc = LoggingConfig()
    # turn on verbosity if requested
    if options.verbose:
        lc.set_verbose()
    # if there is a log file, send DEBUG-level messages there
    if options.log_file:
        lc.set_log_file(options.log_file, logging.DEBUG)
    # set up the logging
    lc.apply()

Calling :meth:`LoggingConfig.apply` configures both Python standard library
:mod:`logging` and :mod:`structlog` to log in a consistent format to ``stderr``
(and to the log file, if configured), sets up progress bars, routes Python
warnings to the log system, and generally makes sure that LensKit and other code
emitting logs (including your code) will send them to useful places.

For setup with even less code, but less flexibility, you can use the
:func:`basic_logging` function.

Environment Variables and Defaults
----------------------------------

:meth:`LoggingConfig.apply` looks for settings in the following places, in
decreasing order of priority:

1.  The explicitly-called methods (e.g. :meth:`~LoggingConfig.set_verbose`).
2.  Environment variables (see below).
3.  Built-in default: print INFO and higher-level messages to the console.

Several environment variables control logging behavior if explicit configuration
methods are not called, allowing you to configure logging even when your scripts
are run by other tools like DVC.

.. envvar:: LK_LOG_LEVEL

    The log level for console log output.  Defaults to ``INFO``.

.. envvar:: LK_LOG_FILE

    Path to a log file to receive log messages.

.. envvar:: LK_LOG_FILE_LEVEL

    The level for messages going to the log file.  Defaults to the console log
    level; this allows you to send ``DEBUG`` messages to the file while only
    ``INFO`` messages go to the console.

Progress Reporting
~~~~~~~~~~~~~~~~~~

LensKit provides a simple progress bar façade for reporting progress in batch
inference, model training, etc.  See the :mod:`lenskit.logging.progress` modules
for details, if you are implementing components from which you want to report
progress.  The key entry point is to call :func:`item_progress` to create a
progress bar.  LensKit automatically handles routing progress to the appropriate
backend.
