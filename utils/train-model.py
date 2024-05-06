#!/usr/bin/env python3
"""
Train a recommendation model and save it to disk.

Usage:
    test-algo.py [options] [-d DATA] --item-item FILE

Options:
    -v, --verbose
        Enable verbose logging.
    -d DATA, --dataset=DATA
        Train with DATA [default: ml-latest-small].
"""

import logging
import pickle
import sys

from docopt import docopt

from lenskit.algorithms import Recommender
from lenskit.algorithms.item_knn import ItemItem
from lenskit.datasets import MovieLens

_log = logging.getLogger("train-model")


def main(args):
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    data = args["--dataset"]
    _log.info("loading data %s", data)
    ml = MovieLens(f"data/{data}")

    if args["--item-item"]:
        algo = ItemItem(20)
    else:
        _log.error("no algorithm specified")
        sys.exit(2)

    algo = Recommender.adapt(algo)
    _log.info("training algorithm")
    algo.fit(ml.ratings)
    _log.info("training complete")

    file = args["FILE"]
    _log.info("saving to %s", file)
    with open(file, "wb") as f:
        pickle.dump(algo, f, 5)


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
