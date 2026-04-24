set -eo pipefail

data="$TEST_WORK/ml-data"
out="$TEST_WORK/lift.pkl.gz"
recs="$TEST_WORK/recs.json"

run-lenskit data convert --movielens "$ML_TEST_DIR" "$data"
run-lenskit train --config pipelines/lift.toml -o "$out" "$data"

require -f "$out"

run-lenskit recommend -o "$recs" --json -n 10 --related-items "$out" 4299
require -f "$recs"

n_users=$(jq 'length' <$recs)
require "$n_users" == 1

n_items=$(jq '.["200"] | length' <$recs)
require "$n_items" == 10
