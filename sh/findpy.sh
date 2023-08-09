#!/usr/bin/env bash

set -ex

cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )/../" &> /dev/null

find . -type d \( \
        -path './venv' -o \
        -path './.*' -o \
        -path './userdata' -o \
        -path './ui' \
        \) -prune -o \( \
        -name '*.py' -o \
        -name '*.pyi' \
        \) \
    | grep -vF './venv' \
    | grep -vF './.' \
    | grep -vF './userdata' \
    | grep -vF './ui'
