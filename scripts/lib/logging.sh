# FILE: logging.sh
#
# This file provides a logging facility. Based on stdlib.zsh, ported to support bash as well.

: ${LOG_VERBOSE:=0}

if [[ $ZSL_USAGE && $usage_verbose ]]; then
    LOG_VERBOSE=1
fi

_zsl_log_error() {
    echo "$(sty bold red)ERR:$(sty no-bold) ${*}$(sty reset)" >&2
}

_zsl_log_warning() {
    echo "$(sty bold yellow)WRN:$(sty no-bold) ${*}$(sty reset)" >&2
}

_zsl_log_notice() {
    echo "$(sty bold magenta)NTC:$(sty no-bold) ${*}$(sty reset)" >&2
}

_zsl_log_action() {
    echo "$(sty bold)${*}$(sty reset)" >&2
}

_zsl_log_success() {
    echo "${_zsl_log_symbol:-💃🏻} $(sty bold green)${*}$(sty reset)" >&2
}

_zsl_log_info() {
    echo "$(sty bold cyan)MSG:$(sty reset) ${*}" >&2
}

_zsl_log_debug() {
    if (($LOG_VERBOSE > 0)); then
        echo "$(sty cyan)DBG:$(sty default) $(sty dim)${*}$(sty reset)" >&2
    fi
}

_zsl_log_trace() {
    if (($LOG_VERBOSE > 1)); then
        echo "$(sty cyan)TRC:$(sty default) $(sty dim)${*}$(sty reset)" >&2
    fi
}

_zsl_log_step() {
    echo "▶  $*"
}

_zsl_msg() {
    local lcmd=_zsl_log_info
    local _zsl_log_symbol=

    while [[ $1 ]]; do
        case "$1" in
        -err | -error)
            lcmd=_zsl_log_error
            shift
            ;;
        -warn | -warning)
            lcmd=_zsl_log_warning
            shift
            ;;
        -dbg | -debug)
            lcmd=_zsl_log_debug
            shift
            ;;
        -trace)
            lcmd=_zsl_log_trace
            shift
            ;;
        -success)
            lcmd=_zsl_log_success
            shift
            ;;
        -action)
            lcmd=_zsl_log_action
            shift
            ;;
        -notice)
            lcmd=_zsl_log_notice
            shift
            ;;
        -symbol)
            _zsl_log_symbol="$2"
            shift 2
            ;;
        -step)
            lcmd=_zsl_log_step
            shift
            ;;
        -- | -info)
            shift
            ;;
        -*)
            _zsl_log_error "invalid msg type $1"
            shift
            ;;
        *)
            break
            ;;
        esac
    done

    "$lcmd" "$@"
}

msg() {
    _zsl_msg "$@"
}

die() {
    local ec=1
    if [[ $1 =~ "^-[0-9]+$" ]]; then
        ec="${1#-}"
        shift
    elif [[ $1 = -* && -v _status_codes[${1#-}] ]]; then
        ec="${_status_codes[${1#-}]}"
        shift
    elif [[ $1 = -* ]]; then
        msg -warn "unknown exit code: ${1#-}"
        shift
    fi
    _zsl_msg -error "$*"
    exit $ec
}
