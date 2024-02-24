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


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the scattermind CLI.

    Returns:
        argparse.Namespace: The argument parser.
    """
    parser = argparse.ArgumentParser(description="Run a scattermind command.")
    subparser = parser.add_subparsers(title="Commands")

    def run_worker(args: argparse.Namespace) -> None:
        worker_start(
            config_file=args.config,
            graph_def=args.graph)

    subparser_worker = subparser.add_parser("worker")
    subparser_worker.set_defaults(func=run_worker)
    parse_args_worker(subparser_worker)

    parser.add_argument(
        "--config",
        type=str,
        help="json config file")

    return parser.parse_args()
