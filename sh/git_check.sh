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

if output=$(git status --porcelain) && ! [ -z "$output" ]; then
    echo "working copy is not clean" >&2
    exit 1
fi

if ! git diff --exit-code 2>&1 >/dev/null && git diff --cached --exit-code 2>&1 >/dev/null ; then
    echo "working copy is not clean" >&2
    exit 2
fi

if ! git diff-index --quiet HEAD -- ; then
    echo "there are uncommitted files" >&2
    exit 3
fi

if [ ! -z $(git ls-files --other --exclude-standard --directory) ]; then
    echo "there are untracked files"
    exit 4
fi
