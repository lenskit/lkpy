# [getoptions] License: Creative Commons Zero v1.0 Universal
# https://github.com/ko1nksm/getoptions (v3.3.2)
getoptions() {
    _error="" _on=1 _no="" _export="" _plus="" _mode="" _alt="" _rest="" _def=""
    _flags="" _nflags="" _opts="" _help="" _abbr="" _cmds="" _init=@empty IFS=" "
    [ $# -lt 2 ] && set -- "${1:?No parser definition}" -
    [ "$2" = - ] && _def=getoptions_parse

    i="					"
    while eval "_${#i}() { echo \"$i\$@\"; }" && [ "$i" ]; do i=${i#?}; done

    quote() {
        q="$2'" r=""
        while [ "$q" ]; do r="$r${q%%\'*}'\''" && q=${q#*\'}; done
        q="'${r%????}'" && q=${q#\'\'} && q=${q%\'\'}
        eval "$1=\${q:-\"''\"}"
    }
    code() {
        [ "${1#:}" = "$1" ] && c=3 || c=4
        eval "[ ! \${$c:+x} ] || $2 \"\$$c\""
    }
    sw() { sw="$sw${sw:+|}$1"; }
    kv() { eval "${2-}${1%%:*}=\${1#*:}"; }
    loop() { [ $# -gt 1 ] && [ "$2" != -- ]; }

    invoke() { eval '"_$@"'; }
    prehook() { invoke "$@"; }
    for i in setup flag param option disp msg; do
        eval "$i() { prehook $i \"\$@\"; }"
    done

    args() {
        on=$_on no=$_no export=$_export init=$_init _hasarg=$1 && shift
        while loop "$@" && shift; do
            case $1 in
            -?) [ "$_hasarg" ] && _opts="$_opts${1#-}" || _flags="$_flags${1#-}" ;;
            +?) _plus=1 _nflags="$_nflags${1#+}" ;;
            [!-+]*) kv "$1" ;;
            esac
        done
    }
    defvar() {
        case $init in
        @none) : ;;
        @export) code "$1" _0 "export $1" ;;
        @empty) code "$1" _0 "${export:+export }$1=''" ;;
        @unset) code "$1" _0 "unset $1 ||:" "unset OPTARG ||:; ${1#:}" ;;
        *)
            case $init in @*) eval "init=\"=\${${init#@}}\"" ;; esac
            case $init in [!=]*)
                _0 "$init"
                return 0
                ;;
            esac
            quote init "${init#=}"
            code "$1" _0 "${export:+export }$1=$init" "OPTARG=$init; ${1#:}"
            ;;
        esac
    }
    _setup() {
        [ "${1#-}" ] && _rest=$1
        while loop "$@" && shift; do kv "$1" _; done
    }
    _flag() {
        args "" "$@"
        defvar "$@"
    }
    _param() {
        args 1 "$@"
        defvar "$@"
    }
    _option() {
        args 1 "$@"
        defvar "$@"
    }
    _disp() { args "" "$@"; }
    _msg() { args "" _ "$@"; }

    cmd() { _mode=@ _cmds="$_cmds${_cmds:+|}'$1'"; }
    "$@"
    cmd() { :; }
    _0 "${_rest:?}=''"

    _0 "${_def:-$2}() {"
    _1 'OPTIND=$(($#+1))'
    _1 "while OPTARG= && [ \"\${$_rest}\" != x ] && [ \$# -gt 0 ]; do"
    [ "$_abbr" ] && getoptions_abbr "$@"

    args() {
        sw="" validate="" pattern="" counter="" on=$_on no=$_no export=$_export
        while loop "$@" && shift; do
            case $1 in
            --\{no-\}*)
                i=${1#--?no-?}
                sw "'--$i'|'--no-$i'"
                ;;
            --with\{out\}-*)
                i=${1#--*-}
                sw "'--with-$i'|'--without-$i'"
                ;;
            [-+]? | --*) sw "'$1'" ;;
            *) kv "$1" ;;
            esac
        done
        quote on "$on"
        quote no "$no"
    }
    setup() { :; }
    _flag() {
        args "$@"
        [ "$counter" ] && on=1 no=-1 v="\$((\${$1:-0}+\$OPTARG))" || v=""
        _3 "$sw)"
        _4 '[ "${OPTARG:-}" ] && OPTARG=${OPTARG#*\=} && set "noarg" "$1" && break'
        _4 "eval '[ \${OPTARG+x} ] &&:' && OPTARG=$on || OPTARG=$no"
        valid "$1" "${v:-\$OPTARG}"
        _4 ";;"
    }
    _param() {
        args "$@"
        _3 "$sw)"
        _4 '[ $# -le 1 ] && set "required" "$1" && break'
        _4 'OPTARG=$2'
        valid "$1" '$OPTARG'
        _4 "shift ;;"
    }
    _option() {
        args "$@"
        _3 "$sw)"
        _4 'set -- "$1" "$@"'
        _4 '[ ${OPTARG+x} ] && {'
        _5 'case $1 in --no-*|--without-*) set "noarg" "${1%%\=*}"; break; esac'
        _5 '[ "${OPTARG:-}" ] && { shift; OPTARG=$2; } ||' "OPTARG=$on"
        _4 "} || OPTARG=$no"
        valid "$1" '$OPTARG'
        _4 "shift ;;"
    }
    valid() {
        set -- "$validate" "$pattern" "$1" "$2"
        [ "$1" ] && _4 "$1 || { set -- ${1%% *}:\$? \"\$1\" $1; break; }"
        [ "$2" ] && {
            _4 "case \$OPTARG in $2) ;;"
            _5 '*) set "pattern:'"$2"'" "$1"; break'
            _4 "esac"
        }
        code "$3" _4 "${export:+export }$3=\"$4\"" "${3#:}"
    }
    _disp() {
        args "$@"
        _3 "$sw)"
        code "$1" _4 "echo \"\${$1}\"" "${1#:}"
        _4 "exit 0 ;;"
    }
    _msg() { :; }

    [ "$_alt" ] && _2 'case $1 in -[!-]?*) set -- "-$@"; esac'
    _2 'case $1 in'
    _wa() { _4 "eval 'set -- $1' \${1+'\"\$@\"'}"; }
    _op() {
        _3 "$1) OPTARG=\$1; shift"
        _wa '"${OPTARG%"${OPTARG#??}"}" '"$2"'"${OPTARG#??}"'
        _4 "${4:-}$3"
    }
    _3 '--?*=*) OPTARG=$1; shift'
    _wa '"${OPTARG%%\=*}" "${OPTARG#*\=}"'
    _4 ";;"
    _3 "--no-*|--without-*) unset OPTARG ;;"
    [ "$_alt" ] || {
        [ "$_opts" ] && _op "-[$_opts]?*" "" ";;"
        [ ! "$_flags" ] || _op "-[$_flags]?*" - "OPTARG= ;;" \
            'case $2 in --*) set -- "$1" unknown "$2" && '"$_rest=x; esac;"
    }
    [ "$_plus" ] && {
        [ "$_nflags" ] && _op "+[$_nflags]?*" + "unset OPTARG ;;"
        _3 "+*) unset OPTARG ;;"
    }
    _2 "esac"
    _2 'case $1 in'
    "$@"
    rest() {
        _4 'while [ $# -gt 0 ]; do'
        _5 "$_rest=\"\${$_rest}" '\"\${$(($OPTIND-$#))}\""'
        _5 "shift"
        _4 "done"
        _4 "break ;;"
    }
    _3 "--)"
    [ "$_mode" = @ ] || _4 "shift"
    rest
    _3 "[-${_plus:++}]?*)" 'set "unknown" "$1"; break ;;'
    _3 "*)"
    case $_mode in
    @)
        _4 "case \$1 in ${_cmds:-*}) ;;"
        _5 '*) set "notcmd" "$1"; break'
        _4 "esac"
        rest
        ;;
    +) rest ;;
    *) _4 "$_rest=\"\${$_rest}" '\"\${$(($OPTIND-$#))}\""' ;;
    esac
    _2 "esac"
    _2 "shift"
    _1 "done"
    _1 '[ $# -eq 0 ] && { OPTIND=1; unset OPTARG; return 0; }'
    _1 'case $1 in'
    _2 'unknown) set "Unrecognized option: $2" "$@" ;;'
    _2 'noarg) set "Does not allow an argument: $2" "$@" ;;'
    _2 'required) set "Requires an argument: $2" "$@" ;;'
    _2 'pattern:*) set "Does not match the pattern (${1#*:}): $2" "$@" ;;'
    _2 'notcmd) set "Not a command: $2" "$@" ;;'
    _2 '*) set "Validation error ($1): $2" "$@"'
    _1 "esac"
    [ "$_error" ] && _1 "$_error" '"$@" >&2 || exit $?'
    _1 'echo "$1" >&2'
    _1 "exit 1"
    _0 "}"

    [ "$_help" ] && eval "shift 2; getoptions_help $1 $_help" ${3+'"$@"'}
    [ "$_def" ] && _0 "eval $_def \${1+'\"\$@\"'}; eval set -- \"\${$_rest}\""
    _0 "# Do not execute" # exit 1
}
# [getoptions_abbr] License: Creative Commons Zero v1.0 Universal
# https://github.com/ko1nksm/getoptions (v3.3.2)
getoptions_abbr() {
    abbr() {
        _3 "case '$1' in"
        _4 '"$1") OPTARG=; break ;;'
        _4 '$1*) OPTARG="$OPTARG '"$1"'"'
        _3 "esac"
    }
    args() {
        abbr=1
        shift
        [ $# -gt 0 ] || return 0
        for i in "$@"; do
            case $i in
            --) break ;;
            [!-+]*) eval "${i%%:*}=\${i#*:}" ;;
            esac
        done
        [ "$abbr" ] || return 0

        for i in "$@"; do
            case $i in
            --) break ;;
            --\{no-\}*)
                abbr "--${i#--\{no-\}}"
                abbr "--no-${i#--\{no-\}}"
                ;;
            --*) abbr "$i" ;;
            esac
        done
    }
    setup() { :; }
    for i in flag param option disp; do
        eval "_$i()" '{ args "$@"; }'
    done
    msg() { :; }
    _2 'set -- "${1%%\=*}" "${1#*\=}" "$@"'
    [ "$_alt" ] && _2 'case $1 in -[!-]?*) set -- "-$@"; esac'
    _2 'while [ ${#1} -gt 2 ]; do'
    _3 'case $1 in (*[!a-zA-Z0-9_-]*) break; esac'
    "$@"
    _3 "break"
    _2 "done"
    _2 'case ${OPTARG# } in'
    _3 "*\ *)"
    _4 'eval "set -- $OPTARG $1 $OPTARG"'
    _4 'OPTIND=$((($#+1)/2)) OPTARG=$1; shift'
    _4 'while [ $# -gt "$OPTIND" ]; do OPTARG="$OPTARG, $1"; shift; done'
    _4 'set "Ambiguous option: $1 (could be $OPTARG)" ambiguous "$@"'
    [ "$_error" ] && _4 "$_error" '"$@" >&2 || exit $?'
    _4 'echo "$1" >&2'
    _4 "exit 1 ;;"
    _3 "?*)"
    _4 '[ "$2" = "$3" ] || OPTARG="$OPTARG=$2"'
    _4 "shift 3; eval 'set -- \"\${OPTARG# }\"' \${1+'\"\$@\"'}; OPTARG= ;;"
    _3 "*) shift 2"
    _2 "esac"
}
# [getoptions_help] License: Creative Commons Zero v1.0 Universal
# https://github.com/ko1nksm/getoptions (v3.3.2)
getoptions_help() {
    _width="30,12" _plus="" _leading="  "

    pad() {
        p=$2
        while [ ${#p} -lt "$3" ]; do p="$p "; done
        eval "$1=\$p"
    }
    kv() { eval "${2-}${1%%:*}=\${1#*:}"; }
    sw() {
        pad sw "$sw${sw:+, }" "$1"
        sw="$sw$2"
    }

    args() {
        _type=$1 var=${2%% *} sw="" label="" hidden="" && shift 2
        while [ $# -gt 0 ] && i=$1 && shift && [ "$i" != -- ]; do
            case $i in
            --*) sw $((${_plus:+4} + 4)) "$i" ;;
            -?) sw 0 "$i" ;;
            +?) [ ! "$_plus" ] || sw 4 "$i" ;;
            *)
                [ "$_type" = setup ] && kv "$i" _
                kv "$i"
                ;;
            esac
        done
        [ "$hidden" ] && return 0 || len=${_width%,*}

        [ "$label" ] || case $_type in
        setup | msg) label="" len=0 ;;
        flag | disp) label="$sw " ;;
        param) label="$sw $var " ;;
        option) label="${sw}[=$var] " ;;
        esac
        [ "$_type" = cmd ] && label=${label:-$var } len=${_width#*,}
        pad label "${label:+$_leading}$label" "$len"
        [ ${#label} -le "$len" ] && [ $# -gt 0 ] && label="$label$1" && shift
        echo "$label"
        pad label "" "$len"
        [ $# -gt 0 ] || return 0
        for i in "$@"; do echo "$label$i"; done
    }

    for i in setup flag param option disp "msg -" cmd; do
        eval "${i% *}() { args $i \"\$@\"; }"
    done

    echo "$2() {"
    echo "cat<<'GETOPTIONSHERE'"
    "$@"
    echo "GETOPTIONSHERE"
    echo "}"
}
