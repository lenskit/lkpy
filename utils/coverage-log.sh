#!/bin/zsh

git fetch --all || exit 2
coverage json || exit 2

if [[ -n $GITHUB_HEAD_REF ]]; then
    # PR run — save difference to output
    git log --pretty=ref $GITHUB_HEAD_REF | while read commit msg; do
        prev_data="$(git notes --ref=coverage show $log)"
        if [[ "$?" -eq 0 ]]; then
            break
        fi
    done
    if [[ -z "$prev_data" ]]; then
        echo "no previous coverage found" >&2
        exit 2
    fi

    prev_cov="$(echo "prev_data" |jq .totals.percent_covered)"
    if [[ $? -ne 0 ]]; then
        echo "jq failed" >&2
        exit 2
    fi
    cur_cov="$(jq .totals.percent_covered coverage.json)"
    if [[ $? -ne 0 ]]; then
        echo "jq failed" >&2
        exit 2
    fi

    cov_change=$(( $cur_cov - $prev_cov ))
    echo "coverage change: $cov_change"
    cat >>"$GITHUB_JOB_SUMMARY" <<EOM
Coverage change **$cov_change** (from $prev_cov to $cur_cov).

EOM
elif [[ $GITHUB_EVENT_NAME = push && -n $GITHUB_TOKEN ]]; then
    jq '{meta: .meta, totals: .totals}' coverage.json >cov-summary.json || exit 2
    git notes --ref=coverage add -F cov-summary.json HEAD || exit 2
    git push origin refs/notes/coverage || exit 2
fi
