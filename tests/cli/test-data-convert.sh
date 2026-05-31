run-lenskit data convert --movielens "$ML_TEST_DIR" "$TEST_WORK/ml-data"
require -d "$TEST_WORK/ml-data"
require -f "$TEST_WORK/ml-data/schema.json"

name="$(jq -r .name "$TEST_WORK/ml-data/schema.json")"
require "$name" = ml-latest-small

if [[ -f data/australian_users_items.json.gz ]]; then
    run-lenskit data convert --steam data/australian_users_items.json.gz "$TEST_WORK/steam-au-data"
    require -d "$TEST_WORK/steam-au-data"
    require -f "$TEST_WORK/steam-au-data/schema.json"
else
    skip 3
fi
