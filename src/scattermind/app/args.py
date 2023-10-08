import argparse

from scattermind.app.worker import worker_start


def parse_args_worker(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--graph",
        type=str,
        help="graph definition json file")
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
    parser = argparse.ArgumentParser(description="Run a scattermind command.")
    subparser = parser.add_subparsers(title="Commands")

    def run_worker(args: argparse.Namespace) -> None:
        worker_start(
            config_file=args.config,
            graph_def_file=args.graph)

    subparser_worker = subparser.add_parser("worker")
    subparser_worker.set_defaults(func=run_worker)
    parse_args_worker(subparser_worker)

    parser.add_argument(
        "--config",
        type=str,
        help="json config file")

    return parser.parse_args()
