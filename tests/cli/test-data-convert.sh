run-python lenskit data convert --movielens "$ML_TEST_DIR" "$TEST_WORK/ml-data"
require -d "$TEST_WORK/ml-data"
require -f "$TEST_WORK/ml-data/schema.json"

name="$(jq -r .name "$TEST_WORK/ml-data/schema.json")"
require "$name" = ml-latest-small
