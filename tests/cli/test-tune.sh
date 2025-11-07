ML100K="data/ml-100k.zip"
data="$TEST_WORK/ml-data"
train="$TEST_WORK/ml-data.train"
test="$TEST_WORK/ml-data.test.parquet"
out="$TEST_WORK/bias-tune"

if ! which ray >/dev/null; then
    msg "ray not installed, skipping tune test"
    exit 0
fi

if [[ ! -f $ML100K ]]; then
    msg "ML-100K not available, skipping tune test"
    exit 0
fi

begin-suite
run-lenskit data convert --movielens "$ML100K" "$data"
run-lenskit data split --fraction=0.2 --min-train-interactions=5 "$data"
run-lenskit tune -T "$train" -V "$test" --save-pipeline "$TEST_WORK/pipeline.json" \
    --max-points=10 pipelines/bias-search.toml "$out"

require -d "$out"
require -f "$out/result.json"
require -f "$TEST_WORK/pipeline.json"
