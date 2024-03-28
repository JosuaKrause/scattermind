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
"""Parses command line arguments of the scattermind CLI."""
import argparse
import json
import sys
from collections.abc import Callable
from typing import cast

from scattermind.api.loader import get_version_info, load_api, VersionInfo
from scattermind.app.healthcheck import perform_healthcheck
from scattermind.app.worker import worker_start
from scattermind.system.config.loader import ConfigJSON


def parse_args_worker(parser: argparse.ArgumentParser) -> None:
    """
    Parse command line arguments for a scattermind worker.

    Args:
        parser (argparse.ArgumentParser): The argument parser.
    """
    parser.add_argument(
        "--graph",
        type=str,
        help="graph definition json file or folder containing json files")
    parser.add_argument(
        "--prefix",  # FIXME implement
        type=str,
        default=None,
        help="an optional fixed prefix for the executor id")
    parser.add_argument(  # FIXME implement
        "--prefix-bonus",
        type=float,
        default=1000.0,
        help="bonus score for nodes with the same executor prefix")


def display_welcome(
        args: argparse.Namespace,
        command: str,
        version_info: VersionInfo | None) -> None:
    """
    Prints the welcome message if `--no-welcome` is unset.

    Args:
        args (argparse.Namespace): The arguments.
        command (str): The name of the command.
        version_info (VersionInfo | None): The optional external version info
            to display.
    """
    if args.no_welcome:
        return
    import scattermind  # pylint: disable=import-outside-toplevel

    print(
        f"Starting {scattermind.__name__}({scattermind.__version__}) "
        f"as {command}")
    print(f"python version: {sys.version}")
    if version_info is not None:
        print(f"{version_info[0]} ({version_info[1]}) on {version_info[2]}")


def parse_args() -> tuple[
        argparse.Namespace,
        Callable[[argparse.Namespace], Callable[[], int | None]]]:
    """
    Parse command line arguments for the scattermind CLI.

    Returns:
        tuple[
                argparse.Namespace,
                Callable[[argparse.Namespace], Callable[[], int | None]]]: A
            tuple of the parsed arguments and the execute function to run.
    """
    parser = argparse.ArgumentParser(description="Run a scattermind command.")
    subparser = parser.add_subparsers(title="Commands")

    def run_worker(args: argparse.Namespace) -> Callable[[], int | None]:
        version_info = get_version_info(args.version_file)
        display_welcome(args, "worker", version_info)
        execute = worker_start(
            config_file=args.config,
            graph_def=args.graph,
            device=args.device,
            version_info=version_info)
        return execute

    subparser_worker = subparser.add_parser("worker")
    subparser_worker.set_defaults(func=run_worker)
    parse_args_worker(subparser_worker)

    def run_healthcheck(args: argparse.Namespace) -> Callable[[], int | None]:
        config_file: str = args.config
        with open(config_file, "rb") as fin:
            config_obj = cast(ConfigJSON, json.load(fin))
        config = load_api(config_obj)

        def execute() -> int | None:
            hc_count = perform_healthcheck(config)
            if hc_count > 0:
                return 0
            print(f"healthcheck has failed with {hc_count}")
            return 1

        return execute

    subparser_hc = subparser.add_parser("healthcheck")
    subparser_hc.set_defaults(func=run_healthcheck)

    parser.add_argument(
        "--config",
        type=str,
        help="json config file")
    parser.add_argument(
        "--version-file",
        type=str,
        default=None,
        help="provide a version file to display external version information")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="overrides the system device")
    parser.add_argument(
        "--no-welcome",
        action="store_true",
        help="suppresses the welcome message")
    parser.add_argument(
        "--boot",
        action="store_true",
        help=(
            "if booting, do not exit on load error. "
            "this prevents crash restart loops"))

    args = parser.parse_args()
    return args, args.func
