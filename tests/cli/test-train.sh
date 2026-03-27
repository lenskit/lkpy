data="$TEST_WORK/ml-data"
out="$TEST_WORK/als.pkl.gz"

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
