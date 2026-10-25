#!/usr/bin/env bash

. "$(dirname "$0")/lib/init.sh" || exit 2

_cli_definition() {
    setup REST help:_cli_help -- "Usage: scripts/build-dist.sh [options]" ''
    msg -- "Options:"
    flag LOG_VERBOSE init:@export -v --verbose -- "verbose log output"
    flag SDIST_ONLY init:=0 -s --sdist -- "build source distribution only"
    flag CLEAN init:=0 -c --clean -- "clean staged sources before building"
    flag DYNAMIC_VERSION init:=0 -d --dynamic-version -- "create dynamically-versioned sdist"
    disp :_cli_help -h --help -- "show help"
}
eval "$(getoptions _cli_definition _cli_parse) exit 1"
_cli_parse "$@"
eval "set -- $REST"
if (($#)); then
    die "unexpected arguments: $*"
fi

git-is-clean() {
    if [[ -z "$(git status -u --porcelain)" ]]; then
        return 0
    else
        return 1
    fi
}

declare -a build_args=()
STAGE_SOURCE=build/staged-source

if (($DYNAMIC_VERSION)); then
    if ! git-is-clean; then
        msg -error "dynamically-versioned build requires fully-committed source"
        git status || true
        exit 3
    fi

    if (($CLEAN)); then
        msg "cleaning $STAGE_SOURCE"
        rm -rf "$STAGE_SOURCE"
    fi

    msg "staging sources"
    mkdir -p build
    git archive --format=tar --prefix=staged-source/ -o build/source.tar HEAD
    if (($?)); then
        msg -error "source archive failed"
        exit 10
    fi
    tar -C build -xf build/source.tar
    if (($?)); then
        msg -error "extracting source archive failed"
        exit 10
    fi

    msg "updating version"
    run-cmd -check ./scripts/version-tool.py --update "$STAGE_SOURCE"

    msg "entering staged source directory"
    build_args+=(-o "$PWD/dist")
    cd "$STAGE_SOURCE"
fi

if (($SDIST_ONLY)); then
    msg "setting to only build source dist"
    build_args+=("--sdist")
fi

msg "building distribution"
run-cmd -check uv build "${build_args[@]}"
