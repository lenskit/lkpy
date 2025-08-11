#!/usr/bin/env bash
#MISE description="Build source distribution"
#MISE depends=["init-dirs"]

set -eo pipefail

declare -A options
declare -a build_args=()
STAGE_SOURCE=build/staged-source

while getopts cdh arg; do
    options[$arg]="${OPTARG:-1}"
done

if ((${options[h]})); then
    cat <<EOD
mise run build:sdist [-vch]

Options:
    -c      clean staged sources before building
    -d      create dynamically-versioned sdist
    -h      print this help message
EOD
    exit
fi

if ((${options[d]})); then
    if [[ -n "$(git status -u --porcelain)" ]]; then
        echo "dynamically-versioned build requires fully-committed source" >&2
        exit 3
    fi

    if ((${options[c]})); then
        echo "cleaning $STAGE_SOURCE"
        rm -rf "$STAGE_SOURCE"
    fi

    echo "staging sources"
    git archive --format=tar --prefix=staged-source/ HEAD |
        tar -C build -xf -
    if (($?)); then
        echo "source archive failed" >&2
        exit 10
    fi

    echo "updating version"
    mise run version -- --update "$STAGED_SOURCE" || exit 11

    echo "entering staged source directory"
    build_args+=(-o "$PWD/dist")
fi

echo "building sdist"
exec uv build --sdist "${build_args[@]}"
