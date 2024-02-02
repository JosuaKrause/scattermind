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

rm "test-results/results-*.xml" || true
OUT=$("${PYTHON}" -m test split --filepath test-results/results.xml --total-nodes "${TOTAL_NODES}" --node-id "${CUR_NODE_IX}")
IFS=',' read -a FILE_INFO <<< "$OUT"
echo "previous timings: ${FILE_INFO[0]}"
FILES=$(echo "${OUT}" | sed -E 's/^[^,]*,//')
echo "selected tests: ${FILES}"
rm -r "test-results/" || true
RESULT_FNAME="results-${PYTHON_NAME}-${CUR_NODE_IX}.xml" "${MAKE}" pytest FILE="${FILES}"
tail -v -n +1 test-results/*.xml
