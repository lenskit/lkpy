# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

import argparse
import json
from pathlib import Path

from lenskit.logging import LoggingConfig, get_logger, stdout_console

from .pipeline import PipelineConfig
from .settings import LenskitSettings
from .tuning import TuningSpec

_log = get_logger("lenskit.schemas")


def main():
    parser = _arg_parser()
    args = parser.parse_args()
    lc = LoggingConfig()
    if args.verbose:
        lc.set_verbose()
    lc.apply()

    if args.pipeline:
        schema = PipelineConfig.model_json_schema()
    elif args.tuner:
        schema = TuningSpec.model_json_schema()
    elif args.config:
        schema = LenskitSettings.model_json_schema()
    else:  # pragma: nocover
        raise RuntimeError("no schema specified")

    if args.output:
        _log.info("saving schema to %s", args.output)
        with open(args.output, "w") as f:
            json.dump(schema, f, indent=2)
    else:
        console = stdout_console()
        console.print_json(json.dumps(schema))


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a LensKit schema.")

    parser.add_argument("-v", "--verbose", action="store_true", help="enable verbose logging")
    parser.add_argument("-o", "--output", type=Path, metavar="FILE", help="write schema to FILE")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pipeline", action="store_true", help="export pipeline schema")
    group.add_argument("--tuner", action="store_true", help="export tuner schema")
    group.add_argument("--config", action="store_true", help="export config schema")

    return parser


if __name__ == "__main__":
    main()
