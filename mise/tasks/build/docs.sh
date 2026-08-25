#!/bin/bash
#MISE description="Build documentation."
#USAGE flag "-P --preview" help="Run preview server."

cmd=(sphinx-build)
. "$MISE_PROJECT_ROOT/mise/task-functions.sh"

if [[ $usage_preview ]]; then
    cmd=(sphinx-autobuild)
fi

echo-run "${cmd[@]}" docs build/site || die "doc build failed"

msg "building schemas"
mkdir -p build/site/schemas
echo-run python -m lenskit.schemas -o build/site/schemas/config.json --config
echo-run python -m lenskit.schemas -o build/site/schemas/pipeline.json --pipeline
echo-run python -m lenskit.schemas -o build/site/schemas/tuner.json --tuner
