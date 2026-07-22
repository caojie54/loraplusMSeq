"""Generate synthetic text-to-SQL predictions for a trained model."""

from generation_test_utils import parse_generation_args, run_generation


BENCHMARK = "synthetic_text_to_sql"


def main() -> None:
    args = parse_generation_args("Run synthetic text-to-SQL generation.")
    run_generation(args, BENCHMARK)


if __name__ == "__main__":
    main()
