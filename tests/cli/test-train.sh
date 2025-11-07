data="$TEST_WORK/ml-data"
out="$TEST_WORK/als.pkl.gz"

run-lenskit data convert --movielens "$ML_TEST_DIR" "$data"
run-lenskit train --config pipelines/als-explicit.toml -o "$out" "$data"

require -f "$out"
run-python "$TEST_DIR/test-train-verify.py" "$out"
