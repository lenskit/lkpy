#!/usr/bin/env bash
#MISE description="Build source distribution"
#MISE depends=["init-dirs"]
#USAGE flag "-s --sdist" help="build source dist only"
#USAGE flag "-c --clean" help="clean staged sources before building"
#USAGE flag "-d --dynamic-version" help="create dynamically-versioned sdist"

set -eo pipefail

. "$MISE_PROJECT_ROOT/mise/task-functions.sh"

declare -a build_args=()
STAGE_SOURCE=build/staged-source

if [[ $usage_dynamic_version = true ]]; then
    if ! git-is-clean; then
        msg "dynamically-versioned build requires fully-committed source"
        exit 3
    fi

    if [[ $usage_clean = true ]]; then
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

if [[ $usage_sdist = true ]]; then
    msg "setting to only build source dist"
    build_args+=("--sdist")
fi

msg "building distribution"
echo-run uv build "${build_args[@]}"
