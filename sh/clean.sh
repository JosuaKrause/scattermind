#!/usr/bin/env bash

set -ex

cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )/../" &> /dev/null

rm -r test-results/ || true
rm -r coverage/ || true
rm -r plugins/test/ || true
rm -r build/ || true
rm -r dist/ || true
