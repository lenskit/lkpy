set -eo pipefail

data="$TEST_WORK/ml-data"
out="$TEST_WORK/als.pkl.gz"
recs="$TEST_WORK/recs.json"

cat >>"$TEST_WORK/verify.py" <<EOF
import pickle
import sys

from xopen import xopen

from lenskit.als import BiasedMFScorer
from lenskit.pipeline.nodes import ComponentInstanceNode

out_file = sys.argv[1]

with xopen(out_file, "rb") as pf:
    pipe = pickle.load(pf)

node = pipe.node("scorer")
assert isinstance(node, ComponentInstanceNode)
assert isinstance(node.component, BiasedMFScorer)
EOF

run-lenskit data convert --movielens "$ML_TEST_DIR" "$data"
run-lenskit train --config pipelines/als-explicit.toml -o "$out" "$data"

require -f "$out"
run-python "$TEST_WORK/verify.py" "$out"

run-lenskit recommend -o "$recs" --json -n 10 "$out" 200
require -f "$recs"

n_users=$(jq 'length' <$recs)
require "$n_users" == 1

n_items=$(jq '.["200"] | length' <$recs)
require "$n_items" == 10
