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

if [[ -f data/anonymous-msweb.data.gz ]]; then
    run-lenskit data convert --ms-web data/anonymous-msweb.data.gz "$TEST_WORK/msweb"
    require -d "$TEST_WORK/msweb"
    require -f "$TEST_WORK/msweb/schema.json"
else
    skip 3
fi
