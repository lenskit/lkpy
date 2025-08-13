Release Notes
=============

Since 2025, LensKit uses calendar-based versioning and includes the release
notes in the primary documentation.  Older release notes are on GitHub.

.. toctree::
    :maxdepth: 1

    2025
    Older Versions <https://github.com/lenskit/lkpy/releases>

.. _SemCalVer:

SemCalVer
---------

.. _CalVer: https://calver.org/

Since 2025, LensKit uses “semantic calendar versioning”.  That is, it uses
CalVer_, so version numbers are based on the year in which the software was
released.  We specifically use the format **YYYY.MINOR.MICRO**, where MINOR is
bumped with each significant revision release in a given year, and MICRO is used
for small bug-fix or minor enhancement releases.

.. _`semantic versioning`: https://semver.org/

We augment this to be compatible with `semantic versioning`_: for APIs
documented to be semver-stable, we batch up breaking changes to the next
new-year calendar release.  This means that you can use dependency constraints
that assume SemVer (e.g. Python's compatible-version constraint ``~=``) on
LensKit versions, and they won't automatically pull in breaking changes to
stable APIs.  It is rare that breaking changes are so urgent that we cannot wait
until the next year to release them.

This combination does mean that we may sometimes release a minor revision with
the previous year's major version number, if there are breaking changes in
progress but not yet ready for release and we need to release new features or
fixes for the current major version.  This will typically happen early in the
year (except for important bug fixes that need to be backported to previous
stable versions).

.. _stability-levels:

Stability Levels
----------------

SemCalVer guarantees are made in accordance with an API's *stability level*, to
clearly document what users can expect and to give sufficient flexibility to
evolve experimental or in-progress APIs.

Stability levels are defined with respect to two different types of interaction
with an API:

-   The *caller* that calls the function, instantiates the class, etc.
-   An *implementer* that implements an interface, subclasses a class, etc.;
    this is mostly relevant for classes and sometimes methods, not individual
    functions.

.. important::

    Stability guarantees do not take effect until :ref:`2025.1` is released.

There are four stability levels for LensKit classes, functions, etc.:

.. glossary::

    Full stability
        For code at the full stability level, we will avoid breaking changes for
        both callers and (where applicable) implementers until the next annual major
        version bump.  Any methods added to a full-stability base class in a minor
        version will have implementations that work in terms of the
        previously-defined abstract methods. Full stability does **not** guarantee
        that minor versions will not add new methods that may conflict with methods
        added by implementors, so exercise care when adding public (non-underscore)
        methods to subclasses.

    Caller stability
        For code at the caller stability level, we will avoid breaking changes for
        callers until the next major version bump, but may change the code in ways
        that break subclasses or interface re-implementations.

    Testing stability
        Testing-level interfaces are generally stable but may have breaking changes
        in minor releases.  Such changes will be clearly documented in the release
        notes.  This stability level is for code we want people to be able to start
        using freely, but may need to change as we gain experience with the interface.

    Internal stability
    Experimental stability
        Internal and experimental interfaces may change at any time, and breaking
        changes will not necessarily be highlighted as such in the release notes.

.. note::

    If a class or function does not specify a stability level, assume the
    internal/experimental level.


.. _dep-policy:

Dependency Versioning
---------------------

.. _SPEC0: https://scientific-python.org/specs/spec-0000/

LensKit uses SPEC0_, a new standard adopted by many packages in the scientific
Python ecosystem, to decide what versions of Python and core dependencies are
supported.  Our CI environment includes a job that runs the tests on the oldest
supported dependency versions to reduce the risk that we accidentally introduce
a dependency on newer code.

We anchor dependency versions based on when the first release of a year occurs
(under SemCalVer_).  We won't drop support for a core dependency version
mid-stream unless absolutely necessary and clearly-communicated.

As of 2024, these versions are:

*   Python 3.10
*   Pandas 1.5
*   SciPy 1.9.0

We do **not** currently include PyTorch in our SPEC0 version commitments, but
LensKit currently works with PyTorch 2.1 and newer.
