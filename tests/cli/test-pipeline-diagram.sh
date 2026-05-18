set -eo pipefail

out="$TEST_WORK/als-implicit.mmd"

run-lenskit pipeline diagram -o "$out" -c pipelines/als-implicit.toml
require -f "$out"

# make sure it can run
run-command mmdc -i "$out" -o "$TEST_WORK/als-implicit.png"
