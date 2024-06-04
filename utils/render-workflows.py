"""
Render workflows and project template infrastructure.

Usage:
    render-workflows.py [-v]

Options:
    -v, --verbose   verbose logging
"""

import logging
import sys
from pathlib import Path

from docopt import docopt

_log = logging.getLogger("render-workflows")


def main(options):
    init_logging(options)
    dir = Path(".github/workflows")


def init_logging(options):
    level = logging.DEBUG if options["--verbose"] else logging.INFO
    logging.basicConfig(stream=sys.stderr, level=level)


if __name__ == "__main__":
    options = docopt(__doc__)
    main(options)
