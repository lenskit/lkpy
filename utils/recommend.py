#!/usr/bin/env python3
# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Train and recommend with a model for basic timing info.

Usage:
    test-algo.py [options] [-d DATA] MODEL USER...
    test-algo.py [options] [-d DATA] MODEL --random-users=N

Options:
    -v, --verbose
        Enable verbose logging.
    -d DATA, --dataset=DATA
        Train with DATA [default: ml-latest-small].
    -o FILE, --output=FILE
        Write recommendations to FILE.
    -r N, --random-users=N
        Recommend for N random users.
    -N N, --num-recs=N
        Generate N recommendations per user [default: 10].
"""

import logging
import pickle
import sys

import numpy as np
from docopt import docopt

from lenskit import batch
from lenskit.data import load_movielens

_log = logging.getLogger("test-algo")


def main(args):
    level = logging.DEBUG if args["--verbose"] else logging.INFO
    logging.basicConfig(stream=sys.stderr, level=level)

    data = args["--dataset"]
    _log.info("loading data %s", data)
    ml = load_movielens(f"data/{data}")

    _log.info("reading model from %s", args["MODEL"])
    with open(args["MODEL"], "rb") as f:
        algo = pickle.load(f)

    rng = np.random.default_rng()

    if args["--random-users"]:
        n = int(args["--random-users"])
        _log.info("selecting %d random users", n)
        users = rng.choice(ml.users.ids(), n)
    else:
        _log.info("using %d specified users", len(args["USER"]))
        users = [int(u) for u in args["USER"]]

    recs = batch.recommend(algo, users, int(args["--num-recs"]), n_jobs=1)
    _log.info("recommendation complete")

    outf = args["--output"]
    if outf:
        _log.info("saving %d recs to %s", len(recs), outf)
        recs.to_csv(outf, index=False)


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
