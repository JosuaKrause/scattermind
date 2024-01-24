#!/usr/bin/env bash
#
# Scattermind distributes computation of machine learning models.
# Copyright (C) 2024 Josua Krause
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
set -ex

cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )/../" &> /dev/null

MAKE="${MAKE:-make}"
PYTHON="${PYTHON:-python}"
VERSION=$(${MAKE} -s version)

git fetch --tags
if git show-ref --tags "v${VERSION}" --quiet; then
    echo "version ${VERSION} already exists"
    exit 1
fi

FILE_WHL="dist/scattermind-${VERSION}-py3-none-any.whl"
FILE_SRC="dist/scattermind-${VERSION}.tar.gz"

if [ ! -f "${FILE_WHL}" ] || [ ! -f "${FILE_SRC}" ]; then
    ${MAKE} pack
fi

${PYTHON} -m twine upload --repository pypi "${FILE_WHL}" "${FILE_SRC}"
git tag "v${VERSION}"
git push origin "v${VERSION}"
echo "successfully deployed ${VERSION}"
