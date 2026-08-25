# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

"""
Export a LensKit schema.

Usage:
    lenskit.schemas [-v] [-o file] (--pipeline | --tuner | --config)

Options:
    -v, --verbose
        Enable verbose logging.
    -o FILE, --output=FILE
        Output file to write schema to.
    --pipeline
        Export pipeline schema.
    --tuner
        Export tuner schema.
    --config
        Export config schema.
"""

import json

from docopt import docopt

from lenskit.logging import LoggingConfig, get_logger, stdout_console

from .pipeline import PipelineConfig
from .settings import LenskitSettings
from .tuning import TuningSpec

_log = get_logger("lenskit.schemas")


def main(args):
    lc = LoggingConfig()
    if args["--verbose"]:
        lc.set_verbose()
    lc.apply()

    if args["--pipeline"]:
        schema = PipelineConfig.model_json_schema()
    elif args["--tuner"]:
        schema = TuningSpec.model_json_schema()
    else:
        schema = LenskitSettings.model_json_schema()

    if out := args["--output"]:
        _log.info("saving schema to %s", out)
        with open(out, "w") as f:
            json.dump(schema, f, indent=2)
    else:
        console = stdout_console()
        console.print_json(json.dumps(schema))


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
