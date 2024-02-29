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
"""A CLI tool for dividing up tests across executor nodes and merging their
results. This is used by github actions to have multiple runners."""
import argparse

from .mng import merge_results, split_tests


def parse_args_split_tests(parser: argparse.ArgumentParser) -> None:
    """
    Parses arguments for splitting tests across multiple executor nodes.

    Args:
        parser (argparse.ArgumentParser): The argument subparser.
    """
    parser.add_argument(
        "--filepath",
        default="test-results/results.xml",
        type=str,
        help="xml with previous test timings")
    parser.add_argument(
        "--total-nodes",
        type=int,
        help="number of test runner nodes")
    parser.add_argument(
        "--node-id",
        type=int,
        help="id of the current test runner node")


def parse_args_merge_results(parser: argparse.ArgumentParser) -> None:
    """
    Parses arguments for merging test results.

    Args:
        parser (argparse.ArgumentParser): The argument subparser.
    """
    parser.add_argument(
        "--dir",
        default="test-results",
        type=str,
        help=(
            "base test result folder. needs subfolder 'parts' "
            "containing all the xml files to join."))
    parser.add_argument(
        "--out-fname",
        type=str,
        help="output file of combined xml")


def parse_args() -> argparse.Namespace:
    """
    Parses the CLI arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Test Utilities")
    subparser = parser.add_subparsers(title="Commands")

    def run_split_tests(args: argparse.Namespace) -> None:
        split_tests(
            filepath=args.filepath,
            total_nodes=args.total_nodes,
            cur_node=args.node_id)

    subparser_split_tests = subparser.add_parser("split")
    subparser_split_tests.set_defaults(func=run_split_tests)
    parse_args_split_tests(subparser_split_tests)

    def run_merge_results(args: argparse.Namespace) -> None:
        merge_results(
            base_folder=args.dir,
            out_filename=args.out_fname)

    subparser_merge_results = subparser.add_parser("merge")
    subparser_merge_results.set_defaults(func=run_merge_results)
    parse_args_merge_results(subparser_merge_results)
    return parser.parse_args()


def run() -> None:
    """Run the CLI tool."""
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    run()
