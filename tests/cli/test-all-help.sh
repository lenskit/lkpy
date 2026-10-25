# Test that all commands can run with --help

CLIST="$TEST_WORK/cmds.list"

msg "listing LensKit commands"
lenskit --list-commands >"$CLIST" || exit 2

declare -a CMDS=()

msg "reading LensKit command list"
while read lk cmd; do
    CMDS+=("$cmd")
done <"$CLIST"

msg "read ${#CMDS[@]} commands"
test-plan "${#CMDS[@]}"

for cmd in "${CMDS[@]}"; do
    run-lenskit $cmd --help
done
