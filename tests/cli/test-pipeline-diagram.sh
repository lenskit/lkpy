out="$TEST_WORK/als-implicit.mmd"

MERMAID_IMG=ghcr.io/mermaid-js/mermaid-cli/mermaid-cli
if [[ -z "$MERMAID" ]]; then
    if [[ -n $MERMAID_DOCKER ]]; then
        dbg "searching for docker"
        if which -s docker; then
            dbg "docker found"
            MERMAID=docker
        else
            warn "MERMAID_DOCKER set but docker not found"
        fi
    elif which -s mmdc; then
        dbg "found Mermaid CLI"
        MERMAID=cli
    fi
fi

run-lenskit pipeline diagram -o "$out" -c pipelines/als-implicit.toml
require -f "$out"

# make sure it can run / render
cd "$TEST_WORK" || exit 2
declare -a mermaid_cmd
if [[ $MERMAID = mmdc ]]; then
    mermaid_cmd=(mmdc)
elif [[ $MERMAID = docker ]]; then
    mermaid_cmd=(docker run --rm -v "$TEST_WORK:/data" "$MERMAID_IMG")
else
    warn "no mermaid found"
    skip 2 "mermaid not found"
    exit 0
fi
run-command "${mermaid_cmd[@]}" -i als-implicit.mmd -o als-explicit.png
require -f als-explicit.png
