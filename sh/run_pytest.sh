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
RESULT_FNAME="${RESULT_FNAME:-results.xml}"
IFS=',' read -a FILE_INFO <<< "$1"
FILES=("${FILE_INFO[@]}")
export USER_FILEPATH=./userdata

coverage erase

${MAKE} compileall

run_test() {
    ${PYTHON} -m pytest \
        -xvv --full-trace \
        --junitxml="test-results/parts/result${2}.xml" \
        --cov --cov-append \
        $1
}
export -f run_test

if ! [ -z "${FILES}" ]; then
    IDX=0
    echo "${FILES[@]}"
    for CUR_TEST in "${FILES[@]}"; do
        run_test $CUR_TEST $IDX
        IDX=$((IDX+1))
    done
else
    IDX=0
    for CUR in $(find 'test' -type d \( \
            -path 'test/data' -o \
            -path 'test/__pycache__' \
            \) -prune -o \( \
            -name '*.py' -and \
            -name 'test_*' \
            \) | \
            grep -E '.*\.py' | \
            sort -sf); do
        run_test ${CUR} $IDX
        IDX=$((IDX+1))
    done
fi
${PYTHON} -m test merge --dir test-results --out-fname ${RESULT_FNAME}
rm -r "test-results/parts/" || true

coverage xml -o coverage/reports/xml_report.xml
coverage html -d coverage/reports/html_report
