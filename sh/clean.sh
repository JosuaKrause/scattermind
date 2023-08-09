#!/usr/bin/env bash

set -ex

cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )/../" &> /dev/null

rm -r test-results/ || true
rm -r coverage/ || true
rm -r plugins/test/ || true
rm -r build/ || true
rm -r dist/ || true

find . -type d \( \
        -path './venv' -o \
        -path './.*' -o \
        -path './userdata' -o \
        -path './ui' \
        \) -prune -o \( \
        -type d \
        -name '__pycache__' \
        \) \
    | grep -vF './venv' \
    | grep -vF './.' \
    | grep -vF './userdata' \
    | grep -vF './ui' \
    | xargs rm -r

rm -r dist build src/scattermind.egg-info || echo "no files to delete"
