# initialization for LensKit scripts, loading libraries

if [[ -n "$BASH_SOURCE" ]]; then
    LK_SCRIPT_LIB="$(dirname "${BASH_SOURCE[0]}")"
elif [[ -n "$ZSH_VERSION" ]]; then
    LK_SCRIPT_LIB="$(dirname "$0")"
else
    echo "cannot locate script library" >&2
    exit 2
fi

. "$LK_SCRIPT_LIB/styles.sh"
. "$LK_SCRIPT_LIB/logging.sh"

PROJECT_ROOT="$(realpath "$LK_SCRIPT_LIB/../..")"
if [[ ! -f "$PROJECT_ROOT/pyproject.toml" ]]; then
    die "script in unexpected location, cannot find LensKit root"
fi

. "$LK_SCRIPT_LIB/run-cmd.sh"
. "$LK_SCRIPT_LIB/getoptions.sh"
