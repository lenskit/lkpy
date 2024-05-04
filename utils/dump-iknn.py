"""
Train and save the item-item similarity matrix.

Usage:
    dump-iknn.py [-d DATA] [-n NBRS] [-m NBRS] [-s SIM] -o FILE

Options:
    -d DATA, --dataset=DATA
        Learn k-NN matrix on DATA [default: ml-latest-small].
    -o FILE, --output=FILE
        Write output to FILE.
"""

import logging
import sys

import pandas as pd
from docopt import docopt

from lenskit.algorithms.item_knn import ItemItem
from lenskit.datasets import MovieLens

_log = logging.getLogger("dump-iknn")


def main(args):
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    data = args["--dataset"]
    _log.info("loading data %s", data)
    ml = MovieLens(f"data/{data}")

    ii_args = {}
    if args["-n"]:
        ii_args["save_nbrs"] = int(args["-n"])
    if args["-m"]:
        ii_args["min_nbrs"] = int(args["-m"])
    if args["-s"]:
        ii_args["min_sim"] = float(args["-s"])

    algo = ItemItem(20, **ii_args)
    _log.info("training algorithm")
    algo.fit(ml.ratings)

    outf = args["--output"]
    _log.info("saving neighbors to %s", outf)
    items = algo.item_index_
    mat = algo.sim_matrix_.to_scipy().tocoo()
    sims = pd.DataFrame({"i1": items[mat.row], "i2": items[mat.col], "sim": mat.data})
    sims.sort_values(["i1", "i2"], inplace=True)
    sims.to_parquet(outf, index=False)


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
