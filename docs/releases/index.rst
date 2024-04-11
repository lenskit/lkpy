Release Notes
=============

Since 2024, LensKit uses calendar-based versioning and includes the release
notes in the primary documentation.  Older release notes are on GitHub.

.. toctree::
    :maxdepth: 1

    2024
    Older Versions <https://github.com/lenskit/lkpy/releases>

.. _SemCalVer:

SemCalVer
---------

.. warning::
    This is the draft policy intended for the first 2024.x release of LensKit.  It is subject
    to change until the official release of 2024.1.

.. _CalVer: https://calver.org/

Since 2024, LensKit uses “semantic calendar versioning”.  That is, it uses CalVer_, so version
numbers are based on the year in which the software was released.  We specifically use the
format **YYYY.MINOR.MICRO**, where MINOR is bumped with each significant revision release in
a given year, and MICRO is used for small bug-fix or minor enhancement releases.

.. _`semantic versioning`: https://semver.org/

We augment this to be compatible with `semantic versioning`_: for APIs documented
to be semver-stable, we batch up breaking changes to the next new-year calendar
release.  This means that you can use dependency constraints that assume SemVer (e.g.
Python's compatible-version constraint ``~=``) on LensKit versions, and they won't
automatically pull in breaking changes to stable APIs.  It is rare that breaking changes
are so urgent that we cannot wait until the next year to release them.

This combination does mean that we may sometimes release a minor revision with the previous
year's major version number, if there are breaking changes in progress but not yet ready for
release and we need to release new features or fixes for the current major version.  This
will typically happen early in the year (except for important bug fixes that need to be backported
to previous stable versions).
