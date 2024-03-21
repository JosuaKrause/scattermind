# Copyright (C) 2024 Josua Krause
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runs the scattermind application."""


def run() -> None:
    """
    Parses the command line arguments and runs the corresponding app.
    """
    # pylint: disable=import-outside-toplevel
    import argparse
    import sys
    from collections.abc import Callable

    from scattermind.app.args import parse_args

    args = parse_args()
    func: Callable[[argparse.Namespace], int | None] = args.func
    ret = func(args)
    if ret is None:
        return
    sys.exit(ret)


if __name__ == "__main__":
    run()
