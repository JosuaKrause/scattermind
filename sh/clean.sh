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

rm -r test-results/ || true
rm -r coverage/ || true
rm -r plugins/test/ || true
rm -r build/ || true
rm -r dist/ || true

find . -type d \( \
        -path './venv' -o \
        -path './.*' -o \
        -path './userdata' \
        \) -prune -o \( \
        -type d \
        -name '__pycache__' \
        \) \
    | grep -vF './venv' \
    | grep -vF './.' \
    | grep -vF './userdata' \
    | xargs --no-run-if-empty rm -r

rm -r src/scattermind.egg-info || echo "no files to delete"

if command -v redis-cli &> /dev/null; then
    redis-cli -p 6380 \
        "EVAL" \
        "for _,k in ipairs(redis.call('keys', KEYS[1])) do redis.call('del', k) end" \
        1 \
        'test:salt:*' \
        || echo "no redis server active"
fi
