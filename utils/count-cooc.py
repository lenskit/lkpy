# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Test co-occurrance counts.

Usage:
    cooc.py [-v] [--ordered] DATA

Options:
    -v, --verbose
        Enable verbose logging.
    --ordered
        Compute ordered co-occurrences.
    DATA
        Data set to load.
"""

from docopt import docopt
from humanize import metric

from lenskit.data import Dataset
from lenskit.logging import LoggingConfig, Stopwatch, get_logger

_log = get_logger("cooc")


def main():
    opts = docopt(__doc__ or "")
    lc = LoggingConfig()
    if opts["--verbose"]:
        lc.set_verbose(True)
    lc.apply()

    data = Dataset.load(opts["DATA"])

    rels = data.interactions()

    _log.info("computing co-occurrances for %s items", metric(data.item_count))
    timer = Stopwatch()
    cooc = rels.co_occurrences("item", order="timestamp" if opts["--ordered"] else None)
    _log.info("computed %s co-occurrances in %s", metric(cooc.nnz), timer)


if __name__ == "__main__":
    main()
