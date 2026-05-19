set -eo pipefail

out="$TEST_WORK/als-implicit.json"
recs="$TEST_WORK/recs.json"

run-lenskit pipeline expand -o "$out" -c pipelines/als-implicit.toml

scorer="$(jq -r .components.scorer.code <"$out")"

require "$scorer" = lenskit.als:ImplicitMFScorer
