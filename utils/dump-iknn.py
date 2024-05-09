# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University
# Copyright (C) 2023-2024 Drexel University
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Train and save the item-item similarity matrix.

Usage:
    dump-iknn.py [-d DATA] [-n NBRS] [-m NBRS] [-s SIM] [-S FILE] [-I FILE]

Options:
    -d DATA, --dataset=DATA
        Learn k-NN matrix on DATA [default: ml-latest-small].
    -S FILE, --sim-output=FILE
        Write similarities to FILE.
    -I FILE, --item-output=FILE
        Write item data to FILE.
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

    i_outf = args["--item-output"]
    _log.info("saving items to %s", i_outf)
    items = algo.item_index_
    stats = pd.DataFrame(
        {"mean": algo.item_means_.numpy(), "nnbrs": algo.item_counts_.numpy()}, index=items
    )
    stats.index.name = "item"
    stats = stats.reset_index()
    stats.to_parquet(i_outf, index=False)

    sim_outf = args["--sim-output"]
    _log.info("saving neighbors to %s", sim_outf)
    mat = algo.sim_matrix_.to_sparse_coo()
    sims = pd.DataFrame(
        {
            "i1": items[mat.indices()[0].numpy()],
            "i2": items[mat.indices()[1].numpy()],
            "sim": mat.values().numpy(),
        }
    )
    sims.sort_values(["i1", "i2"], inplace=True)
    sims.to_parquet(sim_outf, index=False)


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
