"""Generate MedMCQA predictions for a trained model."""

from generation_test_utils import parse_generation_args, run_generation


BENCHMARK = "medmcqa"


def main() -> None:
    args = parse_generation_args("Run MedMCQA generation.")
    run_generation(args, BENCHMARK)


if __name__ == "__main__":
    main()
