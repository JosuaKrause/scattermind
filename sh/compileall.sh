#!/usr/bin/env bash

set -ex

cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )/../" &> /dev/null

PYTHON="${PYTHON:-python}"

find . -type d \( \
        -path './venv' -o \
        -path './.*' -o \
        -path './userdata' -o \
        -path './stubs_pre' -o \
        -path './ui' \
        \) -prune -o \( \
        -type d \
        -name '__pycache__' \
        \) \
    | grep -vF './venv' \
    | grep -vF './.' \
    | grep -vF './userdata' \
    | grep -vF './stubs_pre' \
    | grep -vF './ui' \
    | xargs rm -r

./sh/findpy.sh \
    | xargs ${PYTHON} -m compileall -q -j 0
