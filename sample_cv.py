"""Backward-compatible wrapper for CV sampling CLI."""

from evaluation.cli.sample_cv import main, parse_args

__all__ = ["parse_args", "main"]


if __name__ == "__main__":
    main()
