#!/bin/bash
#MISE description="Update LensKit bibliography."

set -e
BBT_URL="http://127.0.0.1:23119/better-bibtex/export?/group;name:LensKit/collection/LensKit%20References.bibtex"

curl -fL -o docs/lenskit.bib "$BBT_URL"
