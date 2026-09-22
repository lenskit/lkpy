_sty_reset='0'
_sty_bold='1'
_sty_dim='2'
_sty_italic='3'
_sty_underline='4'
_sty_reverse='7'
_sty_normal='22;23;27'
_sty_no_bold='22'
_sty_no_reverse='27'
_sty_no_italic='23'

_sty_black='30'
_sty_bright_black='90'
_sty_red='31'
_sty_bright_red='91'
_sty_green='32'
_sty_bright_green='92'
_sty_yellow='33'
_sty_bright_yellow='93'
_sty_blue='34'
_sty_bright_blue='94'
_sty_magenta='35'
_sty_bright_magenta='95'
_sty_cyan='36'
_sty_bright_cyan='96'
_sty_white='37'
_sty_bright_white='97'
_sty_default='39'

_sty_bg_black='40'
_sty_bg_bright_black='100'
_sty_bg_red='41'
_sty_bg_bright_red='101'
_sty_bg_green='42'
_sty_bg_bright_green='102'
_sty_bg_yellow='43'
_sty_bg_bright_yellow='103'
_sty_bg_blue='44'
_sty_bg_bright_blue='104'
_sty_bg_magenta='45'
_sty_bg_bright_magenta='105'
_sty_bg_cyan='46'
_sty_bg_bright_cyan='106'
_sty_bg_white='47'
_sty_bg_bright_white='107'
_sty_bg_default='49'

_zsl_want_style() {
    if [[ $NO_COLOR ]]; then
        return 1
    elif ((${FORCE_COLOR:-0})); then
        return 0
    elif [[ -t 2 ]]; then
        return 0
    else
        return 1
    fi
}

if [[ -z "$ZSH_VERSION" ]]; then
    . "$LK_SCRIPT_LIB/style-funcs.sh"
else
    . "$LK_SCRIPT_LIB/style-funcs.zsh"
fi
