#!/usr/bin/env bash

set -ex

cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )/../" &> /dev/null

PYTHON="${PYTHON:-python}"

${PYTHON} -m pip install --progress-bar off --upgrade setuptools twine wheel
rm -r dist build src/scatterbrain.egg-info || echo "no files to delete"
${PYTHON} setup.py sdist bdist_wheel
