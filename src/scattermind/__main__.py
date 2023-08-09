

def run() -> None:
    from scattermind.app.args import parse_args

    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    run()
