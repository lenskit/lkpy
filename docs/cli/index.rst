.. _cli:

LensKit CLI
===========

.. program:: lenskit

LensKit comes with a command-line tool, ``lenskit``, that provides a growing set
of operations for LensKit data and pipelines.

The LensKit CLI works through subcommands (like ``git``).

Base Options
~~~~~~~~~~~~

These options can be provided before subcommand to control output.

.. option:: -v, --verbose

    Output DEBUG log messages.

.. option:: --help

    Show command-line tool help messages.


LensKit Commands
~~~~~~~~~~~~~~~~

The following are the supported commands.  Run any with ``--help`` to print its
options and usage.

``lenskit data convert``
    Convert data into LensKit native format.

``lenskit data describe``
    Describe a :class:`~lenskit.data.Dataset`.

``lenskit data fetch``
    Fetch a public dataset.

``lenskit doctor``
    Print information about the installed Python and LensKit environment.

``lenskit version``
    Display the LensKit version.
