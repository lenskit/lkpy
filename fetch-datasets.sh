#!/bin/bash
set -e

fetch_movielens()
{
    testfile="$1"; shift 1
    zipfile="$1"; shift 1

    echo "checking for $testfile"
    if [ ! -r "$testfile" ]; then
        echo "downloading $zipfile"
        wget --no-verbose -O $zipfile http://files.grouplens.org/datasets/movielens/$zipfile
        unzip $zipfile
    fi
}

mkdir -p data
cd data

while [ -n "$1" ]; do
    case "$1" in
        ml-100k) fetch_movielens "ml-100k/u.data" "ml-100k.zip"; shift 1;;
        ml-1m) fetch_movielens "ml-1m/ratings.dat" "ml-1m.zip"; shift 1;;
        ml-10m) fetch_movielens "ml-10M100K/ratings.dat" "ml-10m.zip"; shift 1;;
        ml-20m) fetch_movielens "ml-20m/ratings.csv" "ml-20m.zip"; shift 1;;
        *) echo "Unknown data set $1" >&2; exit 2;;
    esac
done
