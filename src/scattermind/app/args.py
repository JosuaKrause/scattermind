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

from scattermind.app.worker import worker_start


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


def display_welcome(args: argparse.Namespace, command: str) -> None:
    """
    Prints the welcome message if `--no-welcome` is unset.

    Args:
        args (argparse.Namespace): The arguments.
        command (str): The name of the command.
    """
    if args.no_welcome:
        return
    import scattermind  # pylint: disable=import-outside-toplevel

    print(
        f"Starting {scattermind.__name__}({scattermind.__version__}) "
        f"as {command}")


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the scattermind CLI.

    Returns:
        argparse.Namespace: The arguments.
    """
    parser = argparse.ArgumentParser(description="Run a scattermind command.")
    subparser = parser.add_subparsers(title="Commands")

    def run_worker(args: argparse.Namespace) -> None:
        display_welcome(args, "worker")
        worker_start(
            config_file=args.config,
            graph_def=args.graph,
            device=args.device)

    subparser_worker = subparser.add_parser("worker")
    subparser_worker.set_defaults(func=run_worker)
    parse_args_worker(subparser_worker)

    parser.add_argument(
        "--config",
        type=str,
        help="json config file")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="overrides the system device")
    parser.add_argument(
        "--no-welcome",
        action="store_true",
        help="suppresses the welcome message")

    return parser.parse_args()
