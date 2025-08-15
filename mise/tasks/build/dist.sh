#!/usr/bin/env bash
#MISE description="Build source distribution"
#MISE depends=["init-dirs"]

set -eo pipefail

. "$MISE_PROJECT_ROOT/mise/task-functions.sh"

declare -A options
declare -a build_args=()
STAGE_SOURCE=build/staged-source

while getopts cdhs arg; do
    options[$arg]="${OPTARG:-1}"
done

if ((${options[h]})); then
    cat <<EOD
mise run build:sdist [-scdh]

Options:
    -s      build source dist only
    -c      clean staged sources before building
    -d      create dynamically-versioned sdist
    -h      print this help message
EOD
    exit
fi

if ((${options[d]})); then
    if ! git-is-clean; then
        msg "dynamically-versioned build requires fully-committed source"
        exit 3
    fi

    if ((${options[c]})); then
        msg "cleaning $STAGE_SOURCE"
        rm -rf "$STAGE_SOURCE"
    fi

    msg "staging sources"
    mkdir -p build
    git archive --format=tar --prefix=staged-source/ HEAD |
        tar -C build -xf -
    if (($?)); then
        msg "source archive failed"
        exit 10
    fi

    msg "updating version"
    mise run version -- --update "$STAGE_SOURCE" || exit 11

    msg "entering staged source directory"
    build_args+=(-o "$PWD/dist")
    cd "$STAGE_SOURCE"
fi

if ((${options[s]})); then
    msg "setting to only build source dist"
    build_args+=("--sdist")
fi

msg "building distribution"
echo-run uv build "${build_args[@]}"
