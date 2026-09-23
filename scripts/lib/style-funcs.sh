sty() {
    local esc=$'\e['
    local x var code
    if _zsl_want_style; then
        while (($#)); do
            x="${1//-/_}"
            var="_sty_${x}"
            code="${!var}"
            echo -n "${esc}${code}m"
            shift
        done
    fi
}
