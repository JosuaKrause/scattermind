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
    import sys
    import time
    import traceback

    from scattermind.app.args import parse_args
    from scattermind.app.healthcheck import stop_healthcheck

    def handle_error(*, loop: bool) -> None:
        if loop:
            print("Error on boot!")
        else:
            print("Fatal error!")
        print(traceback.format_exc())
        sys.stderr.flush()
        sys.stdout.flush()
        time.sleep(10)
        stop_healthcheck()
        while loop:
            time.sleep(60)

    is_boot = "--boot" in sys.argv
    start_time = time.monotonic()
    try:
        args, func = parse_args()
        execute = func(args)
    except BaseException:  # pylint: disable=broad-except
        if is_boot:
            handle_error(loop=True)
        raise
    try:
        ret = execute()
    except BaseException:  # pylint: disable=broad-except
        if is_boot:
            handle_error(loop=time.monotonic() - start_time < 60.0)
        raise
    sys.stderr.flush()
    sys.stdout.flush()
    if is_boot:
        time.sleep(10.0)
    if ret is None:
        return
    sys.exit(ret)


if __name__ == "__main__":
    run()
