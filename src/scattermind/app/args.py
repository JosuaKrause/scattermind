import argparse

from scattermind.app.worker import worker_start


def parse_args_worker(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--graph",
        type=str,
        help="graph definition json file")


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
