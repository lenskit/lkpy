run-cmd() {
    local echo_info=1 check_rv=0 rv use_sudo=0
    while [[ $1 ]]; do
        case "$1" in
        -dbg-echo)
            echo_info=0
            shift
            ;;
        -check)
            check_rv=1
            shift
            ;;
        -sudo)
            use_sudo=1
            shift
            ;;
        --)
            shift
            break
            ;;
        -*)
            die "invalid option $1"
            ;;
        *)
            break
            ;;
        esac
    done
    local cmd="$1"
    shift

    if (($use_sudo)); then
        if (($echo_info || $LOG_VERBOSE)); then
            echo "+ $(sty bold red)${cmd}$(sty reset) $*" >&2
        fi
        sudo "$cmd" "$@"
        rv="$?"
    else
        if (($echo_info || $LOG_VERBOSE)); then
            echo "+ $(sty bold)${cmd}$(sty no-bold) $*" >&2
        fi
        "$cmd" "$@"
        rv="$?"
    fi
    if (($check_rv && $rv)); then
        msg -error "command ${cmd} exited with code $rv"
        exit $rv
    fi
    return $rv
}
