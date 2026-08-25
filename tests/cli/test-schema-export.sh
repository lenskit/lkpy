test-plan 12

run-python -m lenskit.schemas --pipeline -o $TEST_WORK/pipeline.json
require -f $TEST_WORK/pipeline.json
title=$(jq -r .title <$TEST_WORK/pipeline.json)
if (($?)); then
    not_ok "pipeline.json is invalid JSON"
else
    ok "parsed pipeline.json"
fi
require $title == PipelineConfig

run-python -m lenskit.schemas --tuner -o $TEST_WORK/tuner.json
require -f $TEST_WORK/tuner.json
title=$(jq -r .title <$TEST_WORK/tuner.json)
if (($?)); then
    not_ok "tuner.json is invalid JSON"
else
    ok "parsed tuner.json"
fi
require $title == TuningSpec

run-python -m lenskit.schemas --config -o $TEST_WORK/config.json
require -f $TEST_WORK/config.json
title=$(jq -r .title <$TEST_WORK/config.json)
if (($?)); then
    not_ok "config.json is invalid JSON"
else
    ok "parsed config.json"
fi
require $title == LenskitSettings
